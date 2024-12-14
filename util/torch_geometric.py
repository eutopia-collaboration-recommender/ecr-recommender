import pytest
import torch

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
