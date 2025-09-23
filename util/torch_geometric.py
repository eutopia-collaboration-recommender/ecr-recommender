import pandas as pd
import pytest
import torch

from typing import Tuple
from torch import Tensor
from torch_geometric.metrics.link_pred import LinkPredMetric

class LinkPredHitRate(LinkPredMetric):
    higher_is_better: bool = True

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        return pred_isin_mat.sum(dim=-1) >= 1


def test_hit_rate():
    pred_mat = torch.tensor([[1, 0], [1, 2], [0, 2], [0, 1]])
    edge_label_index = torch.tensor([[0, 0, 2, 2, 3], [0, 1, 2, 1, 2]])

    metric = LinkPredHitRate(k=2)
    metric.update(pred_mat, edge_label_index)
    result = metric.compute()

    assert result == pytest.approx((2) / 3)


def get_mapper_to_contiguous_ids(node_df: pd.DataFrame, id_column: str) -> Tuple[dict, dict]:
    # Get unique nodes
    unique_nodes = node_df[id_column].unique()
    # Create a mapping from node IDs to contiguous IDs
    node_id_map = {node: i for i, node in enumerate(unique_nodes)}
    # Create a mapping from contiguous IDs to node IDs
    id_map = {y: x for x, y in node_id_map.items()}

    # Return the mappings
    return node_id_map, id_map


def get_results(epoch: int,
                train_loss: float,
                test_loss: float,
                evaluation_results: dict,
                verbose: bool = True,
                log_every_n_epochs: int = 1,
                bootstrap_id: int = None) -> dict:
    # Save results
    epoch_result = {
        'Epoch': epoch,
        'Train Loss': train_loss,
        'Test Loss': test_loss,
        'Precision@k': float(evaluation_results['precision@k'].compute()),
        'Recall@k': float(evaluation_results['recall@k'].compute()),
        'MAP@k': float(evaluation_results['map@k'].compute()),
        'MRR@k': float(evaluation_results['mrr@k'].compute()),
        'NDCG@k': float(evaluation_results['ndcg@k'].compute()),
        'HitRate@k': float(evaluation_results['hit_rate@k'].compute())
    }

    if bootstrap_id is not None:
        epoch_result['Bootstrap ID'] = bootstrap_id

    # Print results
    if verbose and epoch % log_every_n_epochs == 0:
        print(
            f"Epoch {epoch}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, precision@k: {epoch_result['Precision@k']:.4f}, recall@k: {epoch_result['Recall@k']:.4f}, MAP@k: {epoch_result['MAP@k']:.4f}, MRR@k: {epoch_result['MRR@k']:.4f}, NDCG@k: {epoch_result['NDCG@k']:.4f}, HitRate@k: {epoch_result['HitRate@k']:.4f}")

    return epoch_result
