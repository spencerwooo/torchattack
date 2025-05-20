from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Type, Union

import torch
import torch.nn as nn

from torchattack.attack_model import AttackModel


class AttackCategory(Enum):
    COMMON = 'COMMON'  # Common attacks that should work on any model
    GRADIENT_VIT = 'GRADIENT_VIT'  # Gradient-based attacks that only work on ViTs
    GENERATIVE = 'GENERATIVE'  # Generative adversarial attacks
    NON_EPS = 'NON_EPS'  # Attacks that do not accept epsilon as a parameter

    @classmethod
    def verify(cls, obj: Union[str, 'AttackCategory']) -> 'AttackCategory':
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + '.', '')]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f'Invalid AttackCategory class provided; expected {cls.__name__} '
                    f'but received {obj.__class__.__name__}.'
                )
        return obj


ATTACK_REGISTRY: dict[str, Type['Attack']] = {}


def register_attack(
    name: str | None = None, category: str | AttackCategory = AttackCategory.COMMON
) -> Callable[[Type['Attack']], Type['Attack']]:
    """Decorator to register an attack class in the attack registry."""

    def wrapper(attack_cls: Type['Attack']) -> Type['Attack']:
        key = name if name else attack_cls.__name__
        if key in ATTACK_REGISTRY:
            return ATTACK_REGISTRY[key]
        attack_cls.attack_name = key
        attack_cls.attack_category = AttackCategory.verify(category)
        ATTACK_REGISTRY[key] = attack_cls
        return attack_cls

    return wrapper


class Attack(ABC):
    """The base class for all attacks."""

    attack_name: str
    attack_category: AttackCategory

    def __init__(
        self,
        model: nn.Module | AttackModel | None,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None,
    ) -> None:
        super().__init__()

        self.model = (
            # If model is an AttackModel, use the model attribute
            model.model
            if isinstance(model, AttackModel)
            # If model is a nn.Module, use the model itself
            else model
            if model is not None
            # Otherwise, use an empty nn.Sequential acting as a dummy model
            else nn.Sequential()
        )

        # Set device to given or defaults to cuda if available
        is_cuda = torch.cuda.is_available()
        self.device = device if device else torch.device('cuda' if is_cuda else 'cpu')

        # If normalize is None, use identity function
        self.normalize = normalize if normalize else lambda x: x

    @classmethod
    def is_category(cls, category: str | AttackCategory) -> bool:
        """Check if the attack class belongs to the given category."""
        return cls.attack_category is AttackCategory.verify(category)

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        name = self.__class__.__name__

        def repr_map(k: str, v: Any) -> str:
            if isinstance(v, float):
                return f'{k}={v:.3f}'
            if k in [
                'model',
                'normalize',
                'feature_module',
                'feature_maps',
                'features',
                'hooks',
                'generator',
            ]:
                return f'{k}={v.__class__.__name__}'
            if isinstance(v, torch.Tensor):
                return f'{k}={v.shape}'
            return f'{k}={v}'

        args = ', '.join(repr_map(k, v) for k, v in self.__dict__.items())
        return f'{name}({args})'

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Attack):
            return False

        eq_name_attrs = [
            'model',
            'normalize',
            'lossfn',
            'feature_module',  # FIA, ILPD, NAA
            'feature_maps',
            'features',
            'hooks',  # PNAPatchOut, TGR, VDC
            'sub_basis',  # GeoDA
            'generator',  # BIA, CDA, LTP
        ]
        for attr in eq_name_attrs:
            if not (hasattr(self, attr) and hasattr(other, attr)):
                continue
            if (
                getattr(self, attr).__class__.__name__
                != getattr(other, attr).__class__.__name__
            ):
                return False

        for attr in self.__dict__:
            if attr in eq_name_attrs:
                continue
            self_val = getattr(self, attr)
            other_val = getattr(other, attr)

            if isinstance(self_val, torch.Tensor):
                if not isinstance(other_val, torch.Tensor):
                    return False
                if not torch.equal(self_val, other_val):
                    return False
            elif self_val != other_val:
                return False

        return True
