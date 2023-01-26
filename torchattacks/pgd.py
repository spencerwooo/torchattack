from typing import Callable

import torch

from torchattacks.base import Attack


class PGD(Attack):
    def __init__(self, transform: Callable, device: torch.device | None) -> None:
        super().__init__(transform, device)
