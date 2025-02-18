import pytest
import torch

from torchattack.evaluate import FoolingRateMetric


@pytest.fixture()
def metric():
    return FoolingRateMetric()


@pytest.fixture()
def labels():
    return torch.tensor([0, 1, 2])


@pytest.fixture()
def clean_logits():
    return torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])


@pytest.fixture()
def adv_logits():
    return torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.9, 0.1, 0.0]])


def test_update(metric, labels, clean_logits, adv_logits):
    metric.update(labels, clean_logits, adv_logits)

    assert metric.total_count.item() == 3
    assert metric.clean_count.item() == 3  # all clean samples are correctly classified
    assert metric.adv_count.item() == 1  # only the 2nd sample is correctly classified


def test_compute(metric, labels, clean_logits, adv_logits):
    metric.update(labels, clean_logits, adv_logits)
    clean_acc, adv_acc, fooling_rate = metric.compute()

    assert clean_acc.item() == pytest.approx(3 / 3)
    assert adv_acc.item() == pytest.approx(1 / 3)
    # fooling_rate = (clean_count - adv_count) / clean_count
    assert fooling_rate.item() == pytest.approx((3 - 1) / 3)


def test_reset(metric):
    metric.reset()
    assert metric.total_count.item() == 0
    assert metric.clean_count.item() == 0
    assert metric.adv_count.item() == 0
