from typing import Callable

import torch
import torch.nn as nn

from torchattack.base import Attack


class SSP(Attack):
    def __init__(
        self,
        model: nn.Module,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None = None,
    ):
        super().__init__(normalize, device)

        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


if __name__ == '__main__':
    from torchattack.runner import run_attack

    run_attack(SSP, attack_cfg={})
