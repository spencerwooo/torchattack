from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn

from torchattack.attack_model import AttackModel


class Attack(ABC):
    """The base class for all attacks."""

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None,
    ) -> None:
        super().__init__()

        # If model is an AttackModel, use the model attribute
        self.model = model.model if isinstance(model, AttackModel) else model

        # Set device to given or defaults to cuda if available
        is_cuda = torch.cuda.is_available()
        self.device = device if device else torch.device('cuda' if is_cuda else 'cpu')

        # If normalize is None, use identity function
        self.normalize = normalize if normalize else lambda x: x

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        name = self.__class__.__name__

        def repr_map(k, v):
            if isinstance(v, float):
                return f'{k}={v:.3f}'
            if k in ['model', 'normalize', 'feature_layer', 'hooks']:
                return f'{k}={v.__class__.__name__}'
            if isinstance(v, torch.Tensor):
                return f'{k}={v.shape}'
            return f'{k}={v}'

        args = ', '.join(repr_map(k, v) for k, v in self.__dict__.items())
        return f'{name}({args})'

    def __eq__(self, other):
        if not isinstance(other, Attack):
            return False

        eq_name_attrs = [
            'model',
            'normalize',
            'lossfn',
            'feature_layer',  # FIA
            'hooks',  # PNAPatchOut, TGR, VDC
            'perceptual_criteria',  # SSP
            'sub_basis',  # GeoDA
        ]
        for attr in eq_name_attrs:
            if not (hasattr(self, attr) and hasattr(other, attr)):
                continue
            if (
                getattr(self, attr).__class__.__name__
                != getattr(other, attr).__class__.__name__
            ):
                return False

        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__dict__
            if attr not in eq_name_attrs
        )
