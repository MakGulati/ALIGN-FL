import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List
import ray
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.common.logger import configure
from flwr.common.typing import Scalar
from ..strategy.FedMoonVisualise import FedMoonVisualise
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import math
import copy
from ..utils.basic_utils import (
    load_weights,
    save_weights,
    get_weights,
    non_iid_train_iid_test_less_samples,
    alignment_dataloader,
    set_params_return,
    CustomDataset,
)
from ..utils.gen_utils import (
    visualize_gen_image,
    visualize_latent_space,
    compute_FID_and_IS,
    make_module_dp,
    get_noise_multiplier,
    compute_epoch_lipschitz_constant,
    evaluate_vae_encoder_split,
    standard_local_model_train_moon,
)
from ..utils.gen_utils import (
    VAE,
    SimpleVAE,
)
from ..utils.utils_server import sample_and_visualize, set_seed

import os
import numpy as np
import ray

EPSILON = 10
DELTA = 1e-5
EPOCHS = 10
BATCH_SIZE = 128
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")


parser.add_argument(
    "--num_rounds", "-r", type=int, default=10, help="Number of FL rounds."
)
parser.add_argument(
    "--lambda_lip", "-lip", type=float, default=0, help="lambda for lip penalty."
)
parser.add_argument(
    "--identifier", "-i", type=str, required=True, help="Name of experiment."
)
parser.add_argument(
    "--latent_dim", "-dim", type=int, required=True, help="Latent dimension."
)
parser.add_argument(
    "--num_clients", "-c", type=int, required=True, help="Number of clients."
)
parser.add_argument("--data_type", "-dt", type=str, default="mnist", help="Dataset.")
parser.add_argument(
    "--dp_type",
    "-dp",
    type=str,
    help=" which kind of gradient clipping dp: full_dp, dec_dp, no_dp.",
    required=True,
)
parser.add_argument(
    "--clt_net_type", "-cnt", type=str, default="vae", help="Client net VAE or GAN."
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.5,
    help="Temperature for contrastive loss",
)
parser.add_argument(
    "--mu",
    type=float,
    default=1.0,
    help="Coefficient for contrastive loss",
)
parser.add_argument(
    "--buffer_size",
    type=int,
    default=5,
    help="Size of the model buffer for contrastive learning",
)


import wandb

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LATENT_DIM = args.latent_dim
IDENTIFIER = args.identifier
NUM_CLIENTS = args.num_clients
NUM_CLASSES = 2 * args.num_clients
IDENTIFIER_FOLDER = f"moon_fl/{IDENTIFIER}"
if not os.path.exists(IDENTIFIER_FOLDER):
    os.makedirs(IDENTIFIER_FOLDER)


configure(identifier=IDENTIFIER, filename=f"{IDENTIFIER_FOLDER}/training_logs.log")
set_seed(42)
# Download dataset and partition it
if args.num_clients == 2:
    _, valsets, _ = non_iid_train_iid_test_less_samples(
        class_partitions=[(0, 1), (2, 3)],
        data_type=args.data_type,
        outlier_count=600,
        samples_per_class=6000,
    )
    trainsets, testsets, outlier_dataset = non_iid_train_iid_test_less_samples(
        class_partitions=[(0, 1), (2, 3)],
        data_type=args.data_type,
        outlier_count=600,
        samples_per_class=6000,
    )
elif args.num_clients == 5:
    _, valsets, _ = non_iid_train_iid_test_less_samples(
        class_partitions=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
        data_type=args.data_type,
        outlier_count=600,
        samples_per_class=6000,
    )
    trainsets, testsets, outlier_dataset = non_iid_train_iid_test_less_samples(
        class_partitions=[
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
        ],
        data_type=args.data_type,
        outlier_count=600,
        samples_per_class=6000,
    )


class dpSimpleVAE(SimpleVAE):
    def __init__(self, input_dim, latent_dim):
        super(dpSimpleVAE, self).__init__(input_dim, latent_dim)
        make_module_dp(self.encoder)
        make_module_dp(self.decoder)
        make_module_dp(self.fc_mu)
        make_module_dp(self.fc_logvar)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class only_dec_dp_SimpleVAE(SimpleVAE):
    def __init__(self, input_dim, latent_dim):
        super(only_dec_dp_SimpleVAE, self).__init__(input_dim, latent_dim)

        make_module_dp(self.decoder)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, noise_multiplier, cid):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid
        self.noise_multiplier = noise_multiplier
        self.prev_models = []  # Store previous model versions for contrastive learning

        # Create client-specific folder for model storage
        self.models_folder = f"{IDENTIFIER_FOLDER}/client_{self.cid}_models"
        os.makedirs(self.models_folder, exist_ok=True)

        # Load previous models if they exist
        self._load_prev_models()

        # Instantiate model
        if args.clt_net_type == "vae" and args.dp_type == "full_dp":
            self.model = dpSimpleVAE(
                input_dim=784,
                latent_dim=LATENT_DIM,
            )
        elif args.clt_net_type == "vae" and args.dp_type == "dec_dp":
            self.model = only_dec_dp_SimpleVAE(
                input_dim=784,
                latent_dim=LATENT_DIM,
            )
        elif args.clt_net_type == "vae" and args.dp_type == "no_dp":
            self.model = SimpleVAE(
                input_dim=784,
                latent_dim=LATENT_DIM,
            )

        self.global_model = SimpleVAE(
            input_dim=784,
            latent_dim=LATENT_DIM,
        )

        # Determine device
        self.device = DEVICE
        self.model.to(DEVICE)

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        # Update local model with global parameters
        set_params(self.model, parameters)

        # Update global model with the same parameters for contrastive learning
        self.global_model = set_params_return(self.global_model, parameters)

        # Add client ID to config
        config["cid"] = self.cid
        config["temperature"] = args.temperature
        config["mu"] = args.mu
        config["folder"] = IDENTIFIER_FOLDER
        print(len(self.prev_models))
        # Train the model using MOON approach
        all_losses, trained_model = standard_local_model_train_moon(
            self.trainset, self.model, config, None, self.prev_models
        )
        assert trained_model == self.model, "Model training failed"
        # Save model parameters for debugging
        save_weights(
            get_weights(self.model),
            f"{IDENTIFIER_FOLDER}/model_params_cid_{self.cid}_server_round_{config.get('server_round')}.npz",
        )

        # Set a fixed seed for reproducibility
        seed = 42
        torch.manual_seed(seed)

        # Prepare validation dataset
        combine_valset = ConcatDataset([self.valset, outlier_dataset])
        indices = list(range(len(combine_valset)))
        sampler = SubsetRandomSampler(indices)
        val_dataloader = DataLoader(combine_valset, batch_size=64, sampler=sampler)

        # Visualize latent space
        latent_rep = visualize_latent_space(
            self.model,
            val_dataloader,
            self.device,
            f'for_client_{self.cid}_latent_at_round_{config.get("server_round")}',
            folder=IDENTIFIER_FOLDER,
        )

        # Prepare metrics
        metrics = {
            "cid": self.cid,
            "latent_rep": latent_rep,
            "client_round": config["server_round"],
        }
        metrics.update(all_losses)

        # Store a copy of the current model for future contrastive learning
        # Maintain only the last few models as specified by buffer size
        if len(self.prev_models) >= args.buffer_size:
            self.prev_models.pop(0)  # Remove oldest model

        # Add current model to previous models
        self.prev_models.append(copy.deepcopy(self.model))

        # Save models to disk for persistence between rounds
        self._save_prev_models(config.get("server_round", 0))

        print(
            f"Client {self.cid} now has {len(self.prev_models)} previous models saved"
        )

        # Return updated model parameters and metrics
        return (
            self.get_parameters({}),
            len(self.trainset),
            metrics,
        )

    def _save_prev_models(self, server_round):
        """Save previous models to disk"""
        for idx, model in enumerate(self.prev_models):
            model_path = f"{self.models_folder}/model_round_{server_round}_idx_{idx}.pt"
            torch.save(model.state_dict(), model_path)

        # Save list of model paths for loading
        model_list_path = f"{self.models_folder}/model_list.txt"
        with open(model_list_path, "w") as f:
            for idx in range(len(self.prev_models)):
                f.write(f"model_round_{server_round}_idx_{idx}.pt\n")

    def _load_prev_models(self):
        """Load previous models from disk if they exist"""
        model_list_path = f"{self.models_folder}/model_list.txt"
        if not os.path.exists(model_list_path):
            print(f"No previous models found for client {self.cid}")
            return

        # Clear existing models
        self.prev_models = []

        # Load model list
        with open(model_list_path, "r") as f:
            model_files = f.read().splitlines()

        # Load each model
        for model_file in model_files:
            model_path = f"{self.models_folder}/{model_file}"
            if os.path.exists(model_path):
                # Create a new model instance
                model = SimpleVAE(input_dim=784, latent_dim=LATENT_DIM)
                model.load_state_dict(torch.load(model_path))
                self.prev_models.append(model)

        print(f"Loaded {len(self.prev_models)} previous models for client {self.cid}")

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset."""
        # Update model with global parameters
        set_params(self.model, parameters)

        # Prepare test dataloader
        testloader = DataLoader(self.valset, batch_size=64)
        loss, accuracy, clf_loss = 0, 0, 0

        # Evaluate
        # Return statistics
        return (
            float(loss),
            1,
            {
                "accuracy": float(accuracy),
                "cid": self.cid,
                "local_val_loss": float(loss),
            },
        )


def get_client_fn(train_partitions, test_partitions, noise_multipliers):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        trainset, valset = train_partitions[int(cid)], test_partitions[int(cid)]

        # Create and return client
        return FlowerClient(
            trainset, valset, noise_multipliers[int(cid)], int(cid)
        ).to_client()

    return client_fn


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Main function
def cleanup_client_models():
    """Clean up previous model files from all clients after a run completes"""
    import shutil
    import glob

    print("Cleaning up client model files...")

    # Get all client model directories
    client_dirs = glob.glob(f"{IDENTIFIER_FOLDER}/client_*_models")

    for client_dir in client_dirs:
        if os.path.exists(client_dir):
            print(f"Removing model files from {client_dir}")
            # Option 1: Remove individual model files but keep the directory structure
            model_files = glob.glob(f"{client_dir}/*.pt")
            for file in model_files:
                os.remove(file)

            # Also remove the model list file
            if os.path.exists(f"{client_dir}/model_list.txt"):
                os.remove(f"{client_dir}/model_list.txt")

            # Option 2 (alternative): Remove the entire directory
            # shutil.rmtree(client_dir)

    print("Cleanup completed")


def main():
    # Parse input arguments
    run = wandb.init(
        entity="mak",
        group="moon",
        reinit=True,
    )

    print(f"running these hparams-> {wandb.config}")
    wandb.define_metric("server_round")
    wandb.define_metric("global_*", step_metric="server_round")
    wandb.define_metric("generated_*", step_metric="server_round")
    wandb.define_metric("client_round")
    wandb.define_metric("train_*", step_metric="client_round")
    wandb.define_metric("eval_*", step_metric="client_round")

    samples_per_class = wandb.config["sample_per_class"]

    ALIGNMENT_DATALOADER = None

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "local_epochs": wandb.config["local_epochs"],
            "lambda_kl": wandb.config["lambda_kl"],
            "lambda_lip": wandb.config["lambda_lip"],
            "max_grad_norm": wandb.config["max_grad_norm"],
            "batch_size": wandb.config["batch_size"],
            "temperature": args.temperature,
            "mu": args.mu,
            "server_round": server_round,
            "total_rounds": args.num_rounds,
            "latent_dim": LATENT_DIM,
            "allow_agg": False,
            "num_tasks": NUM_CLIENTS,
            "folder": IDENTIFIER_FOLDER,
        }
        return config

    def eval_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "server_round": server_round,
            "allow_agg": False,
            "batch_size": 64,
            "num_tasks": NUM_CLIENTS,
        }
        return config

    def get_evaluate_fn(testset):
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, Scalar],
        ):
            """Use the entire test set for evaluation."""

            # Determine device
            device = DEVICE
            config["hdim"] = LATENT_DIM
            config["batch_size"] = 64
            config["global_task"] = True

            model = SimpleVAE(
                input_dim=784,
                latent_dim=LATENT_DIM,
            )
            model.to(device)
            set_params(model, parameters)
            model.eval()

            # Set a fixed seed for reproducibility
            seed = 42
            torch.manual_seed(seed)

            global_testset_wth_outlier = ConcatDataset([testset, outlier_dataset])
            # Create a list of indices for the dataset
            indices = list(range(len(global_testset_wth_outlier)))

            # Create a SubsetRandomSampler with the indices
            sampler = SubsetRandomSampler(indices)
            # Create the DataLoader with the sampler
            testloader_wth_ol = DataLoader(
                global_testset_wth_outlier, batch_size=64, sampler=sampler
            )

            testloader_wthout_ol = DataLoader(testset, batch_size=128, shuffle=True)

            true_img, gen_img = visualize_gen_image(
                model,
                testloader_wth_ol,
                device,
                f"server_eval_{server_round}",
                folder=IDENTIFIER_FOLDER,
                batch_size=64,
            )

            latent_reps = visualize_latent_space(
                model,
                testloader_wth_ol,
                device,
                filename=f"server_eval_{server_round}",
                folder=IDENTIFIER_FOLDER,
            )

            global_fid, IS_mean, _ = compute_FID_and_IS(
                testloader_wthout_ol, model, device, None, 32
            )
            lip_cons_model = compute_epoch_lipschitz_constant(
                model, testloader_wthout_ol, device, mode="complete"
            )
            lip_cons_dec = compute_epoch_lipschitz_constant(
                model, testloader_wthout_ol, device, mode="decoder"
            )
            metrics, _ = evaluate_vae_encoder_split(model, testloader_wthout_ol, device)
            latent_reps_wthout_ol = visualize_latent_space(
                model,
                testloader_wthout_ol,
                device,
                filename=f"server_eval_wo_{server_round}",
                folder=IDENTIFIER_FOLDER,
            )
            arbitrary_samples = sample_and_visualize(
                model,
                device,
                64,
                IDENTIFIER_FOLDER,
                server_round,
                LATENT_DIM,
            )

            wandb.log(
                {
                    f"global_true_image": wandb.Image(true_img),
                    f"global_gen_image": wandb.Image(gen_img),
                    f"global_latent_rep": (
                        wandb.Image(latent_reps)
                        if isinstance(latent_reps, str)
                        else latent_reps
                    ),
                    f"global_latent_rep_wo_outlier": (
                        wandb.Image(latent_reps_wthout_ol)
                        if isinstance(latent_reps_wthout_ol, str)
                        else latent_reps_wthout_ol
                    ),
                    f"global_arbitrary_samples": wandb.Image(arbitrary_samples),
                    f"global_lip_cons_model": lip_cons_model,
                    f"global_lip_cons_dec": lip_cons_dec,
                    f"global_fid": global_fid,
                    f"global_accuracy": metrics["accuracy"],
                    f"global_f1": metrics["f1"],
                    f"server_round": server_round,
                },
                step=server_round,
            )

        return evaluate

    # Download dataset and partition it
    if args.dp_type == "full_dp" or args.dp_type == "dec_dp":
        noise_multipliers = []
        for trainset in trainsets:
            sampling_prob = BATCH_SIZE / len(trainset)
            steps = math.ceil(len(trainset) / BATCH_SIZE) * EPOCHS * args.num_rounds
            noise_multiplier = get_noise_multiplier(
                eps=EPSILON, delta=DELTA, steps=steps, sampling_prob=sampling_prob
            )
            noise_multipliers.append(noise_multiplier)
    else:
        noise_multipliers = [0] * len(trainsets)

    initial_vae = SimpleVAE(
        input_dim=784,
        latent_dim=LATENT_DIM,
    )

    initial_params = ndarrays_to_parameters(get_weights(initial_vae))

    strategy = FedMoonVisualise(
        initial_parameters=initial_params,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(valsets[-1]),
        model_buffer_size=args.buffer_size,
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": 36 / NUM_CLIENTS,
        "num_gpus": 1 / NUM_CLIENTS,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(trainsets, testsets, noise_multipliers),
        num_clients=NUM_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args={
            "include_dashboard": True,  # we need this one for tracking
        },
    )

    # Clean up client model files after simulation
    cleanup_client_models()

    ray.shutdown()
    wandb.finish()


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "metric": {"name": "global_accuracy", "goal": "maximize"},
        "parameters": {
            # client config
            "local_epochs": {"values": [EPOCHS]},
            "batch_size": {"values": [BATCH_SIZE]},
            "lambda_kl": {"values": [1]},
            "lambda_lip": {"values": [args.lambda_lip]},
            "sample_per_class": {"values": [20]},
            "max_grad_norm": {"values": [1]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project=IDENTIFIER)

    wandb.agent(sweep_id, function=main, count=1)

    # Clean up client model files after simulation
    cleanup_client_models()
