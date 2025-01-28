from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvg
import wandb


class VisualiseFedAvg(FedAvg):
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
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
