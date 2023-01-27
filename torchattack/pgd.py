from typing import Callable

import torch
import torch.nn as nn

from torchattack.base import Attack


class PGD(Attack):
    """The Projected Gradient Descent (PGD) attack.

    From the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    https://arxiv.org/abs/1706.06083
    """

    def __init__(
        self,
        model: nn.Module,
        transform: Callable,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the PGD attack.

        Args:
            model: The model to attack.
            transform: A transform to normalize images.
            eps: The maximum perturbation. Defaults to 8/255.
            steps: Number of steps. Defaults to 10.
            alpha: Step size, `eps / (steps / 2)` if None. Defaults to None.
            random_start: Start from random uniform perturbation. Defaults to True.
            clip_min: Minimum value for clipping. Defaults to 0.0.
            clip_max: Maximum value for clipping. Defaults to 1.0.
            targeted: Targeted attack if True. Defaults to False.
            device: Device to use for tensors. Defaults to cuda if available.
        """

        super().__init__(transform, device)

        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform PGD on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        # If random start enabled, delta (perturbation) is then randomly
        # initialized with samples from a uniform distribution.
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
            delta.requires_grad_()
        else:
            delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / (steps / 2)
        if self.alpha is None:
            self.alpha = self.eps / (self.steps / 2)

        # Perform PGD
        for _ in range(self.steps):
            # Compute loss
            outs = self.model(self.transform(x + delta))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Update delta
            g = delta.grad.data.sign()

            delta.data = delta.data + self.alpha * g
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.data.zero_()

        return x + delta


if __name__ == "__main__":
    from torchattack.utils import run_attack

    run_attack(PGD, {"eps": 8 / 255, "steps": 20, "random_start": True})
