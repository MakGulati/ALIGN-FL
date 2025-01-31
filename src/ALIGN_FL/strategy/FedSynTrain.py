from typing import Callable, Dict, List, Optional, Tuple, Union

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
    ndarray_to_bytes,
    bytes_to_ndarray,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvg
import wandb

from ..utils.basic_utils import (
    set_params_return,
    get_weights,
    create_subset_data,
)
from torch.utils.data import DataLoader, ConcatDataset
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import torch
from ..utils.utils_server import train_global_vae
import os
import numpy as np
from ..utils.gen_utils import (
    SimpleVAE,
    sample_from_model,
    visualize_gen_image,
    visualize_latent_space,
    SampleDataset,
)
from torchvision.utils import save_image


class Fed_Syntrain(FedAvg):
    """Adapted FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
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
        device=None,
        alignment_dataloader=None,
        folder=None,
        svr_cfg=None,
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
        self.device = device
        self.svr_cfg = svr_cfg
        self.alignment_dataloader = alignment_dataloader
        self.folder = folder
        self.global_model = SimpleVAE(
            input_dim=self.svr_cfg["input_dim"],
            latent_dim=self.svr_cfg["latent_dim"],
        ).to(self.device)
        self.global_params = self.initial_parameters

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"Fed_Sytrain(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        _, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        order_of_clients = [fit_res.metrics["cid"] for _, fit_res in results]
        weighted_results_with_clients = list(zip(weights_results, order_of_clients))

        # Sort weighted_results_with_clients based on the order_of_clients
        weighted_results_with_clients_sorted = sorted(
            weighted_results_with_clients, key=lambda x: x[1]
        )

        # Extract the sorted weights_results
        sorted_weights_results = [
            item[0] for item in weighted_results_with_clients_sorted
        ]
        sorted_weights = [item[0] for item in sorted_weights_results]
        sorted_clients_examples = [x[1] for x in weighted_results_with_clients_sorted]
        if self.svr_cfg["clt_net_type"] == "vae":
            temp_local_models = [
                SimpleVAE(
                    latent_dim=self.svr_cfg["latent_dim"],
                    input_dim=self.svr_cfg["input_dim"],
                ).to(self.device)
                for idx in range(len(weights_results))
            ]

        loaded_local_models = [
            set_params_return(model, sorted_weights[idx])
            for idx, model in enumerate(temp_local_models)
        ]

        fake_datasets = []
        for idx in range(len(loaded_local_models)):
            features, labels = sample_from_model(
                loaded_local_models[idx],
                self.device,
                self.svr_cfg["global_sample_size"],
                idx,
                server_round,
                self.svr_cfg["folder"],
            )
            fake_datasets.append(SampleDataset(features, labels))

        combined_dataset = ConcatDataset(fake_datasets)

        client0_data = fake_datasets[0]
        client0_dataloader = DataLoader(
            client0_data,
            batch_size=20 * 2 * self.svr_cfg["num_tasks"],
            shuffle=True,
        )
        client1_data = fake_datasets[1]
        client1_dataloader = DataLoader(
            client1_data,
            batch_size=20 * 2 * self.svr_cfg["num_tasks"],
            shuffle=True,
        )
        for features, labels in client0_dataloader:
            fake_img = features
            save_image(
                fake_img.view(-1, 1, 28, 28),
                f"{self.folder}/debug_img_clt0_img{server_round}.png",
            )
            break
        for features, labels in client1_dataloader:
            fake_img = features
            save_image(
                fake_img.view(-1, 1, 28, 28),
                f"{self.folder}/debug_img_clt1_img{server_round}.png",
            )
            break

        subset_fake_dataset = create_subset_data(
            combined_dataset, 20 * 2 * self.svr_cfg["num_tasks"]
        )

        subset_fake_dataloader = DataLoader(
            subset_fake_dataset,
            batch_size=20 * 2 * self.svr_cfg["num_tasks"],
            shuffle=True,
        )

        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]

        if self.global_params is not None:
            self.global_model = set_params_return(
                self.global_model, parameters_to_ndarrays(self.global_params)
            )
        print(f"server config: {self.svr_cfg}")
        self.global_model = train_global_vae(
            combined_dataset, self.global_model, self.svr_cfg
        )

        # fake_img, gen_fake_img = visualize_gen_image(
        #     self.global_model,
        #     testloader=subset_fake_dataloader,
        #     device=self.device,
        #     rnd=server_round,
        #     folder=f"{self.folder}/after_server_train_fake",
        #     batch_size=20 * 2 * self.svr_cfg["num_tasks"],
        # )

        # true_img_a, gen_img_a = visualize_gen_image(
        #     self.global_model,
        #     testloader=self.alignment_dataloader,
        #     device=self.device,
        #     rnd=server_round,
        #     folder=f"{self.folder}/after_server_train",
        #     batch_size=20 * 2 * self.svr_cfg["num_tasks"],
        # )

        # latent_a = visualize_latent_space(
        #     model=self.global_model,
        #     test_loader=self.alignment_dataloader,
        #     device=self.device,
        #     filename=f"latent_space_{server_round}",
        #     folder=f"{self.folder}/after_server_train",
        # )

        parameters_aggregated = ndarrays_to_parameters(get_weights(self.global_model))
        self.global_params = parameters_aggregated

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

        # logging the server logs
        server_logs = {}
        # server_logs[f"train_fake_image_used_to_train_server"] = wandb.Image(fake_img)
        # server_logs[f"train_gen_of_fake_image_used_to_train_server"] = wandb.Image(
        #     gen_fake_img
        # )
        # # after
        # server_logs[f"train_true_image_after_global_server_train"] = wandb.Image(
        #     true_img_a
        # )
        # server_logs[f"train_gen_image_after_global_server_train"] = wandb.Image(
        #     gen_img_a
        # )

        # server_logs[f"train_latent_after_global_server_train"] = wandb.Image(latent_a)
        wandb.log(data=server_logs, step=server_round)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]

        for eval_metric in eval_metrics:
            data = {}
            for key, value in eval_metric[1].items():
                if key not in ["cid", "local_gen_sampled_image_arb"]:
                    data[f"eval_{key}_{eval_metric[1]['cid']}"] = value

            wandb.log(data=data, step=server_round)

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
        return loss_aggregated, metrics_aggregated
