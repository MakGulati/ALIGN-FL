from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import deque

import torch
import wandb
import ray
import numpy as np
import os

from flwr.common import (
    EvaluateIns,
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..utils.basic_utils import (
    set_params_return,
    get_weights,
    create_subset_data,
)
from ..utils.gen_utils import (
    SimpleVAE,
    sample_from_model,
    SampleDataset,
)
from ..utils.utils_server import train_global_vae
from torch.utils.data import DataLoader, ConcatDataset, Dataset


class HybridMoonAlignStrategy(FedAvg):
    """
    Hybrid strategy combining MOON's contrastive learning with ALIGN_FL's synthetic data generation.

    This strategy:
    1. Uses a model buffer like MOON to enable contrastive learning on clients
    2. Generates synthetic data on the server like ALIGN_FL to align client models
    3. Performs a global training step on the server using synthetic data
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        model_buffer_size: int = 5,
        device=None,
        server_config: Optional[Dict[str, Scalar]] = None,
        folder: str = None,
        cleanup_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        # Store previous model parameters for contrastive learning (MOON)
        self.model_buffer = deque(maxlen=model_buffer_size)
        if initial_parameters is not None:
            # Add the initial parameters to the model buffer
            self.model_buffer.append(parameters_to_ndarrays(initial_parameters))

        # Server-side model and configuration (ALIGN_FL)
        self.device = (
            device
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.svr_cfg = server_config
        self.folder = folder

        if self.svr_cfg is not None:
            self.global_model = SimpleVAE(
                input_dim=self.svr_cfg["input_dim"],
                latent_dim=self.svr_cfg["latent_dim"],
            ).to(self.device)

        self.cleanup_fn = cleanup_fn

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"HybridMoonAlignStrategy(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results and perform server-side training with synthetic data

        1. Aggregate client parameters with FedAvg (weighted average) if use_fedavg=True
        2. Generate synthetic data from each client model
        3. Train the global model on this synthetic data
        4. Return the improved global model
        """
        # First check if we should use FedAvg or ALIGN_FL style stateful server
        use_fedavg = self.svr_cfg.get("use_fedavg", True)

        if use_fedavg:
            # Aggregate parameters using FedAvg (parent class method)
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(
                server_round, results, failures
            )

            if parameters_aggregated is None:
                return None, {}

            # Add current parameters to the model buffer for contrastive learning
            self.model_buffer.append(parameters_to_ndarrays(parameters_aggregated))
        else:
            # In non-averaging mode, we still need metrics
            metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = {}

            # If it's the first round and we don't have a global model yet,
            # initialize with the first client's parameters
            if server_round == 1 and not hasattr(self, "global_params"):
                self.global_params = results[0][1].parameters

                # Initialize the model buffer for contrastive learning
                self.model_buffer.append(parameters_to_ndarrays(self.global_params))

            # In non-averaging mode, we use the current global model state
            # and don't aggregate client models
            parameters_aggregated = self.global_params

        # Log client metrics to wandb
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        for fit_metric in fit_metrics:  # loop through the clients
            data = {}
            # fit metric is a tuple of num_examples and metrics dict
            for key, value in fit_metric[1].items():
                # retrieved from clients
                if isinstance(value, (int, float)):
                    data[f"train_{key}_{fit_metric[1]['cid']}"] = value
                elif isinstance(value, str):
                    data[f"train_{key}_{fit_metric[1]['cid']}"] = wandb.Image(value)

            wandb.log(data=data, step=server_round)

        # ALIGN_FL part: Generate synthetic data from client models and train global model
        if self.svr_cfg is not None:
            # Convert results to list of weights and number of examples
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            order_of_clients = [fit_res.metrics["cid"] for _, fit_res in results]
            weighted_results_with_clients = list(zip(weights_results, order_of_clients))

            # Sort by client ID for consistency
            weighted_results_with_clients_sorted = sorted(
                weighted_results_with_clients, key=lambda x: x[1]
            )

            # Extract sorted weights
            sorted_weights_results = [
                item[0] for item in weighted_results_with_clients_sorted
            ]
            sorted_weights = [item[0] for item in sorted_weights_results]

            # Create temporary models for each client
            temp_local_models = []
            for _ in range(len(sorted_weights)):
                if self.svr_cfg["clt_net_type"] == "vae":
                    model = SimpleVAE(
                        input_dim=self.svr_cfg["input_dim"],
                        latent_dim=self.svr_cfg["latent_dim"],
                    ).to(self.device)
                    temp_local_models.append(model)

            # Load weights into temporary models
            loaded_local_models = [
                set_params_return(model, sorted_weights[idx])
                for idx, model in enumerate(temp_local_models)
            ]

            # Generate synthetic data from each client model
            fake_datasets = []
            num_samples_per_client = self.svr_cfg.get(
                "synthetic_samples_per_client", 1000
            )

            for idx, model in enumerate(loaded_local_models):
                # Sample from model
                features, labels = sample_from_model(
                    model,
                    self.device,
                    num_samples_per_client,
                    idx,
                    server_round,
                    self.folder,
                )
                # Create dataset
                fake_dataset = SampleDataset(features, labels)
                fake_datasets.append(fake_dataset)

            # Combine all synthetic datasets
            combined_dataset = ConcatDataset(fake_datasets)

            # Take a subset for visualization (optional)
            subset_size = min(100, len(combined_dataset))
            indices = np.random.choice(
                len(combined_dataset), subset_size, replace=False
            )
            subset_fake_dataset = torch.utils.data.Subset(combined_dataset, indices)
            subset_fake_dataloader = DataLoader(
                subset_fake_dataset,
                batch_size=min(20, subset_size),
                shuffle=True,
            )

            # Load model parameters for next steps
            use_fedavg = self.svr_cfg.get("use_fedavg", True)

            if use_fedavg:
                # Load the FedAvg-aggregated parameters into the global model
                self.global_model = set_params_return(
                    self.global_model, parameters_to_ndarrays(parameters_aggregated)
                )
            else:
                # In stateful mode, just use the current global model state
                if hasattr(self, "global_params"):
                    self.global_model = set_params_return(
                        self.global_model, parameters_to_ndarrays(self.global_params)
                    )

            # Train the global model on synthetic data
            print(
                f"Training global model on {len(combined_dataset)} synthetic samples (round {server_round})"
            )
            self.global_model = train_global_vae(
                combined_dataset, self.global_model, self.svr_cfg
            )

            # Get the refined global parameters
            refined_parameters = ndarrays_to_parameters(get_weights(self.global_model))

            # Store the global parameters for stateful updates
            self.global_params = refined_parameters

            # Clean up temporary data if needed
            if self.cleanup_fn:
                self.cleanup_fn()

            # Return the refined parameters after server-side training
            return refined_parameters, metrics_aggregated

        # If server config is not provided, just return the aggregated parameters
        return parameters_aggregated, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with model buffer information."""
        # Get all model history to send to clients
        model_history = list(self.model_buffer)

        # Create client data by extending the config with the model history
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Add the model buffer size to the config
        config["model_buffer_size"] = len(model_history)

        # Default to fraction_fit clients
        num_clients = client_manager.num_available()
        clients_to_fit = max(int(num_clients * self.fraction_fit), self.min_fit_clients)

        # Select clients randomly
        clients = client_manager.sample(
            num_clients=clients_to_fit, min_num_clients=self.min_fit_clients
        )

        # Create and return client instructions
        fit_ins_list = []
        for client in clients:
            # Create the fit instructions with parameters and config
            fit_ins = FitIns(parameters, config.copy())
            fit_ins_list.append((client, fit_ins))

        return fit_ins_list

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        # Call super class method
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Log evaluation metrics
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        for eval_metric in eval_metrics:
            data = {}
            for key, value in eval_metric[1].items():
                if isinstance(value, (int, float)):
                    data[f"eval_{key}_{eval_metric[1]['cid']}"] = value
                elif isinstance(value, str):
                    data[f"eval_{key}_{eval_metric[1]['cid']}"] = wandb.Image(value)

            wandb.log(data=data, step=server_round)

        # Log aggregated metrics
        data_agg = {
            f"eval_{key}_aggregated": value for key, value in metrics_aggregated.items()
        }
        wandb.log(
            data={
                "eval_loss_aggregated": loss_aggregated,
                **data_agg,
            },
            step=server_round,
        )

        # Clean up if needed
        if self.cleanup_fn:
            self.cleanup_fn()

        return loss_aggregated, metrics_aggregated
