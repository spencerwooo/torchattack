from typing import Callable

import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel

EPS_FOR_DIVISION = 1e-12


@register_attack()
class PGDL2(Attack):
    """The Projected Gradient Descent (PGD) attack, with L2 constraint.

    > From the paper: [Towards Deep Learning Models Resistant to Adversarial
    Attacks](https://arxiv.org/abs/1706.06083).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation, measured in L2. Defaults to 1.0.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        random_start: Start from random uniform perturbation. Defaults to True.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        eps: float = 1.0,
        steps: int = 10,
        alpha: float | None = None,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

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
            delta = torch.empty_like(x).normal_()
            delta_flat = delta.reshape(x.size(0), -1)

            n = delta_flat.norm(p=2, dim=1).view(x.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)

            delta = delta * r / n * self.eps
            delta.requires_grad_()

        else:
            delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform PGD
        for _ in range(self.steps):
            # Compute loss
            outs = self.model(self.normalize(x + delta))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Update delta
            g = delta.grad

            g_norms = (
                torch.norm(g.reshape(x.size(0), -1), p=2, dim=1) + EPS_FOR_DIVISION
            )
            g = g / g_norms.view(x.size(0), 1, 1, 1)
            delta.data = delta.data + self.alpha * g

            delta_norms = torch.norm(delta.reshape(x.size(0), -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta.data = delta.data * factor.view(-1, 1, 1, 1)

            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(PGDL2, {'eps': 1.0, 'steps': 10, 'random_start': False})
