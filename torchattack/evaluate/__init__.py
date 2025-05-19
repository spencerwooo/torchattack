from torchattack.evaluate.dataset import NIPSDataset, NIPSLoader
from torchattack.evaluate.meter import FoolingRateMeter
from torchattack.evaluate.runner import run_attack
from torchattack.evaluate.save_image import save_image_batch


class DeprecatedFoolingRateMetric:
    def __new__(cls, *args, **kwargs):
        import warnings

        warnings.warn(
            '`FoolingRateMetric` is deprecated. Use `FoolingRateMeter` instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return FoolingRateMeter(*args, **kwargs)


FoolingRateMetric = DeprecatedFoolingRateMetric

__all__ = [
    'run_attack',
    'save_image_batch',
    'FoolingRateMeter',
    'NIPSDataset',
    'NIPSLoader',
]
