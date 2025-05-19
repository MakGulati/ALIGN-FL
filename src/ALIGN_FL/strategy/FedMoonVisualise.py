from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import deque

import torch
import wandb

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
)


class FedMoonVisualise(FedAvg):
    """MOON (Model-Contrastive Federated Learning) strategy implementation."""

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
        # Store previous model parameters for contrastive learning
        self.model_buffer = deque(maxlen=model_buffer_size)
        if initial_parameters is not None:
            # Add the initial parameters to the model buffer
            self.model_buffer.append(parameters_to_ndarrays(initial_parameters))

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedMoonVisualise(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # Call super method to aggregate parameters and metrics
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Add current parameters to the model buffer
        if parameters_aggregated is not None:
            self.model_buffer.append(parameters_to_ndarrays(parameters_aggregated))

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
                if isinstance(value, (int, float)):
                    data[f"eval_{key}_{eval_metric[1]['cid']}"] = value
                elif isinstance(value, str):
                    data[f"eval_{key}_{eval_metric[1]['cid']}"] = wandb.Image(value)

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

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get all model history to send to clients
        model_history = list(self.model_buffer)

        # Create client data by extending the config with the model history
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Add the model buffer size to the config
        config["model_buffer_size"] = len(model_history)

        # Default to fraction_fit clients if fraction_fit was provided
        num_clients = client_manager.num_available()
        clients_to_fit = int(num_clients * self.fraction_fit)

        # Ensure at least min_fit_clients are selected
        if clients_to_fit < self.min_fit_clients:
            clients_to_fit = self.min_fit_clients

        # Select clients randomly
        clients = client_manager.sample(
            num_clients=clients_to_fit, min_num_clients=self.min_fit_clients
        )

        # Create FitIns for each client
        fit_ins_list = []
        for client in clients:
            # Create copy of config for each client
            client_config = config.copy()

            # Create the fit instructions
            fit_ins = FitIns(parameters, client_config)

            # Append to the list of fit instructions
            fit_ins_list.append((client, fit_ins))

        return fit_ins_list
