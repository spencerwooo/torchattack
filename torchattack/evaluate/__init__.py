from typing import Any

from torchattack.evaluate.dataset import NIPSDataset, NIPSLoader
from torchattack.evaluate.meter import FoolingRateMeter
from torchattack.evaluate.runner import run_attack
from torchattack.evaluate.save_image import save_image_batch


class DeprecatedFoolingRateMetric(FoolingRateMeter):
    """Deprecated class for FoolingRateMetric."""

    def __new__(cls, *args: Any, **kwargs: Any) -> 'DeprecatedFoolingRateMetric':
        import warnings

        warnings.warn(
            '`FoolingRateMetric` is deprecated. Use `FoolingRateMeter` instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls)


FoolingRateMetric = DeprecatedFoolingRateMetric

__all__ = [
    'run_attack',
    'save_image_batch',
    'FoolingRateMeter',
    'NIPSDataset',
    'NIPSLoader',
]
