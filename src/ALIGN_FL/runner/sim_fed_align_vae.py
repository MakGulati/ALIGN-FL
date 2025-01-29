import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import os
import sys
import torch
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, ndarray_to_bytes
from flwr.common.logger import configure
from flwr.common.typing import Scalar
from ..strategy.FedSynTrain import Fed_Syntrain
import math

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
    compute_fid_complete,
    make_module_dp,
    get_noise_multiplier,
    compute_epoch_lipschitz_constant,
    evaluate_vae_encoder_split,
)
from ..utils.gen_utils import (
    VAE,
    local_model_train_only,
    SimpleVAE,
    local_model_train_only_with_dp,
    local_model_train_only_enc_penalty,
)
from ..utils.utils_server import sample_and_visualize, set_seed
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
    "--lambda_lip", "-lip", type=float, default=0.5, help="lambda for lip penalty."
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


import wandb

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LATENT_DIM = args.latent_dim
IDENTIFIER = args.identifier
NUM_CLIENTS = args.num_clients
NUM_CLASSES = 2 * args.num_clients
IDENTIFIER_FOLDER = f"fed_align/{IDENTIFIER}"
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


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, noise_multiplier, cid):
        self.trainset = trainset
        self.valset = valset
        self.cid = cid
        self.noise_multiplier = noise_multiplier

        # Instantiate model
        if args.clt_net_type == "vae" and args.dp_type == "full_dp":
            self.model = dpSimpleVAE(
                input_dim=784,
                latent_dim=LATENT_DIM,
            )
        if args.clt_net_type == "vae" and args.dp_type == "dec_dp":
            self.model = only_dec_dp_SimpleVAE(
                input_dim=784,
                latent_dim=LATENT_DIM,
            )
        if args.clt_net_type == "vae" and args.dp_type == "no_dp":
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
        self.model.to(self.device)
        self.global_model.to(self.device)
        initial_weights = [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]

        save_weights(
            initial_weights,
            f"{IDENTIFIER_FOLDER}/model_params_cid_{self.cid}_server_round_{0}.npz",
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        if config["allow_agg"]:
            set_params(self.model, parameters)
        else:

            loaded_model = load_weights(
                f"{IDENTIFIER_FOLDER}/model_params_cid_{self.cid}_server_round_{config['server_round']-1}.npz"
            )
            set_params(self.model, loaded_model)
        config["noise_multiplier"] = self.noise_multiplier
        print(f"client {self.cid} training and nm={config['noise_multiplier']}")
        config["cid"] = self.cid
        # client training
        if args.dp_type == "full_dp" or args.dp_type == "dec_dp":
            all_losses, trained_model = local_model_train_only_with_dp(
                self.trainset,
                self.model,
                config,
            )
        else:
            all_losses, trained_model = local_model_train_only(
                self.trainset,
                self.model,
                config,
            )
        assert trained_model == self.model, "Model is not the same"
        save_weights(
            get_weights(self.model),
            f"{IDENTIFIER_FOLDER}/model_params_cid_{self.cid}_server_round_{config.get('server_round')}.npz",
        )
        # Set a fixed seed for reproducibility
        seed = 42
        torch.manual_seed(seed)

        combine_valset = ConcatDataset([self.valset, outlier_dataset])

        # Create a list of indices for the dataset
        indices = list(range(len(combine_valset)))

        # Create a SubsetRandomSampler with the indices
        sampler = SubsetRandomSampler(indices)

        val_dataloader = DataLoader(combine_valset, batch_size=64, sampler=sampler)

        # true_img, gen_img = visualize_gen_image(
        #     self.model,
        #     val_dataloader,
        #     self.device,
        #     f'for_client_{self.cid}_train_at_round_{config.get("server_round")}',
        #     folder=IDENTIFIER_FOLDER,
        #     batch_size=64,
        # )
        ft_global_model = set_params_return(self.global_model, parameters)
        # true_img_ft_glb, gen_img_ft_glb = visualize_gen_image(
        #     ft_global_model,
        #     val_dataloader,
        #     self.device,
        #     f'for_client_{self.cid}_global_train_at_round_{config.get("server_round")}',
        #     folder=IDENTIFIER_FOLDER,
        #     batch_size=64,
        # )
        assert ft_global_model != self.model, "Global model is same as local model"
        # latent representation of finutuned global model

        latent_rep = visualize_latent_space(
            self.model,
            val_dataloader,
            self.device,
            f'for_client_{self.cid}_latent_at_round_{config.get("server_round")}',
            folder=IDENTIFIER_FOLDER,
        )

        print(f"client {self.cid} done training")
        print(len(self.trainset))

        metrics = {
            "cid": self.cid,
            # "true_image": true_img,
            # "gen_image": gen_img,
            "latent_rep": latent_rep,
            # "true_image_ft_glb": true_img_ft_glb,
            # "gen_image_ft_glb": gen_img_ft_glb,
            "client_round": config["server_round"],
        }
        metrics.update(all_losses)

        return (
            self.get_parameters({}),
            len(self.trainset),  # placeholder for num_samples
            metrics,
        )

    def evaluate(self, parameters, config):
        # TODO: Implement evaluation
        # Construct dataloader
        # valloader = DataLoader(self.valset, batch_size=64)

        # # Evaluate
        # set_params(self.model, parameters)
        loss, accuracy, clf_loss = 0, 0, 0

        # Return statistics
        return (
            float(loss),
            1,
            {
                # "accuracy": float(accuracy),
                # "cid": self.cid,
                # "clf_loss": clf_loss,
            },
        )


def get_client_fn(train_partitions, test_partitions, noise_multipliers):
    """Return a function to construct a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
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

    weighted_metrics = {
        key: [
            num_examples * m[key]
            for num_examples, m in metrics
            if isinstance(m[key], float)
        ]
        for num_examples, m in metrics
        for key in m
        if isinstance(m[key], float)
    }
    examples = [num_examples for num_examples, _ in metrics]

    return {key: sum(value) / sum(examples) for key, value in weighted_metrics.items()}


def main():
    # Parse input arguments
    run = wandb.init(
        entity="mak",
        group="align-fl",
        reinit=True,
    )

    samples_per_class = wandb.config["sample_per_class"]
    print(f"running these hparams-> {wandb.config}")
    wandb.define_metric("server_round")
    wandb.define_metric("global_*", step_metric="server_round")
    wandb.define_metric("generated_*", step_metric="server_round")
    wandb.define_metric("client_round")
    wandb.define_metric("train_*", step_metric="client_round")
    wandb.define_metric("eval_*", step_metric="client_round")

    ALIGNMENT_DATALOADER = alignment_dataloader(
        samples_per_class=samples_per_class,
        batch_size=samples_per_class * NUM_CLASSES,
        shuffle=True,
        num_classes=NUM_CLASSES,
        data_type=args.data_type,
    )

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "local_epochs": wandb.config["local_epochs"],
            "lambda_kl": wandb.config["lambda_kl"],
            "lambda_lip": wandb.config["lambda_lip"],
            "max_grad_norm": wandb.config["max_grad_norm"],
            "batch_size": wandb.config["batch_size"],
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

    server_config = {
        "latent_dim": LATENT_DIM,
        "num_tasks": NUM_CLIENTS,
        "input_dim": 784,
        "hidden_dim": 256,
        "global_epochs": wandb.config["global_epochs"],
        "global_lambda_kl": wandb.config["global_kl_lambda"],
        "global_batch_size": wandb.config["global_batch_size"],
        "global_sample_size": wandb.config["global_sample_size"],
        "clt_net_type": args.clt_net_type,
        "folder": IDENTIFIER_FOLDER,
    }

    def get_evaluate_fn(testset):
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
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

            global_val_loss, global_val_accu, global_clf_loss = 0, 0, 0
            global_fid = compute_fid_complete(testloader_wthout_ol, model, device, None)
            lip_cons_model = compute_epoch_lipschitz_constant(
                model, testloader_wthout_ol, device, mode="complete"
            )
            lip_cons_dec = compute_epoch_lipschitz_constant(
                model, testloader_wthout_ol, device, mode="decoder"
            )
            metrics, _ = evaluate_vae_encoder_split(model, testloader_wthout_ol, device)

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
                    f"global_arbitrary_samples": wandb.Image(arbitrary_samples),
                    # f"global_val_loss": global_val_loss,
                    # f"global_val_accu": global_val_accu,
                    # f"global_clf_loss": global_clf_loss,
                    f"global_lip_cons_model": lip_cons_model,
                    f"global_lip_cons_dec": lip_cons_dec,
                    f"global_fid": global_fid,
                    f"global_accuracy": metrics["accuracy"],
                    f"global_f1": metrics["f1"],
                    # f"global_psnr": global_psnr,
                    # f"global_ssim": global_ssim,
                    # f"global_fid": global_fid,
                    # f"global_fc": global_fc,
                    f"server_round": server_round,
                },
                step=server_round,
            )

        return evaluate

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
    # initial model for global model
    initial_vae = SimpleVAE(
        input_dim=784,
        latent_dim=LATENT_DIM,
    )

    initial_params = ndarrays_to_parameters(get_weights(initial_vae))
    # trained_pca_model = train_save_pca_model(
    #     initial_vae, ALIGNMENT_DATALOADER, DEVICE, IDENTIFIER_FOLDER
    # )
    strategy = Fed_Syntrain(
        initial_parameters=initial_params,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(valsets[-1]),  # Global evaluation function
        device=DEVICE,
        alignment_dataloader=ALIGNMENT_DATALOADER,
        folder=IDENTIFIER_FOLDER,
        svr_cfg=server_config,
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
    # Clean up .npz and .joblib files from the folder
    for file in os.listdir(IDENTIFIER_FOLDER):
        if file.endswith(".npz") or file.endswith(".joblib"):
            os.remove(os.path.join(IDENTIFIER_FOLDER, file))
    ray.shutdown()
    wandb.finish()


if __name__ == "__main__":

    sweep_config = {
        "method": "grid",
        "metric": {"name": "global_ssim", "goal": "maximize"},
        "parameters": {
            # client config
            "local_epochs": {"values": [EPOCHS]},
            "batch_size": {"values": [BATCH_SIZE]},
            "lambda_kl": {"values": [1]},
            "lambda_lip": {"values": [args.lambda_lip]},
            "sample_per_class": {"values": [20]},
            "max_grad_norm": {"values": [1]},
            # server config
            "global_epochs": {"values": [20]},
            "global_batch_size": {"values": [128]},
            "global_kl_lambda": {"values": [1]},
            "global_sample_size": {"values": [5000]},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project=IDENTIFIER)

    wandb.agent(sweep_id, function=main, count=1)
