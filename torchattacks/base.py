from abc import ABC, abstractmethod
from typing import Any, Optional

import torch


class Attack(ABC):
    """The base class for all attacks."""

    def __init__(self, device: Optional[torch.device]) -> None:
        super().__init__()

        # Set device to given or defaults to cuda if available.
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        name = self.__class__.__name__

        def repr_map(k, v):
            if isinstance(v, float):
                return f"{k}={v:.3f}"
            if k == "model":
                return f"{k}={v.__class__.__name__}"
            return f"{k}={v}"

        args = ", ".join(repr_map(k, v) for k, v in self.__dict__.items())
        return f"{name}({args})"
