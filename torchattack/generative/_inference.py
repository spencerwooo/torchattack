from abc import abstractmethod
from typing import Any

import torch

from torchattack._attack import Attack
from torchattack.generative._weights import WeightsEnum


class GenerativeAttack(Attack):
    def __init__(
        self,
        device: torch.device | None = None,
        eps: float = 10 / 255,
        weights: WeightsEnum | str | None = None,
        checkpoint_path: str | None = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        # Generative attacks do not require specifying model and normalize.
        super().__init__(model=None, normalize=None, device=device)

        self.eps = eps
        self.weights = weights
        self.checkpoint_path = checkpoint_path
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Initialize the generator and its weights
        self.generator = self._init_generator()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform the generative attack via generator inference on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        x_unrestricted = self.generator(x)
        delta = torch.clamp(x_unrestricted - x, -self.eps, self.eps)
        x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        return x_adv

    @abstractmethod
    def _init_generator(self, *args: Any, **kwds: Any) -> Any:
        pass
