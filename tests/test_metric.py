import pytest
import torch

from torchattack.eval import FoolingRateMetric


@pytest.fixture()
def metric():
    return FoolingRateMetric()


def test_initial_state(metric):
    assert metric.total_count.item() == 0
    assert metric.clean_count.item() == 0
    assert metric.adv_count.item() == 0


def test_update(metric):
    labels = torch.tensor([0, 1, 2])
    clean_logits = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    adv_logits = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.9, 0.1, 0.0]])

    metric.update(labels, clean_logits, adv_logits)

    assert metric.total_count.item() == 3
    assert metric.clean_count.item() == 3  # all clean samples are correctly classified
    assert metric.adv_count.item() == 1  # only the 2nd sample is correctly classified


def test_compute(metric):
    labels = torch.tensor([0, 1, 2])
    clean_logits = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    adv_logits = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.9, 0.1, 0.0]])

    metric.update(labels, clean_logits, adv_logits)
    clean_acc, adv_acc, fooling_rate = metric.compute()

    assert clean_acc.item() == pytest.approx(3 / 3)
    assert adv_acc.item() == pytest.approx(1 / 3)
    # fooling_rate = (clean_count - adv_count) / clean_count
    assert fooling_rate.item() == pytest.approx((3 - 1) / 3)
