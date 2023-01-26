from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class Attack(ABC):
    """The base class for all attacks."""

    def __init__(self, transform: Callable, device: torch.device | None) -> None:
        super().__init__()
        self.transform = transform

        # Set device to given or defaults to cuda if available.
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        name = self.__class__.__name__

        def repr_map(k, v):
            if isinstance(v, float):
                return f"{k}={v:.3f}"
            if k in ["model", "transform"]:
                return f"{k}={v.__class__.__name__}"
            return f"{k}={v}"

        args = ", ".join(repr_map(k, v) for k, v in self.__dict__.items())
        return f"{name}({args})"

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any):
        pass
