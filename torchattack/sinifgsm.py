from typing import Callable

import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class SINIFGSM(Attack):
    """The SI-NI-FGSM (Scale-invariant Nesterov-accelerated Iterative FGSM) attack.

    > From the paper: [Nesterov Accelerated Gradient and Scale Invariance for Adversarial
    Attacks](https://arxiv.org/abs/1908.06281).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        m: Number of scaled copies of the image. Defaults to 3.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        m: int = 3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.m = m
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform SI-NI-FGSM on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        g = torch.zeros_like(x)
        delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform SI-NI-FGSM
        for _ in range(self.steps):
            # Nesterov gradient component
            nes = self.alpha * self.decay * g
            x_nes = x + delta + nes

            # Gradient is computed over scaled copies
            grad = torch.zeros_like(x)

            # Obtain scaled copies of the images
            for i in torch.arange(self.m):
                x_ness = x_nes / torch.pow(2, i)

                # Compute loss
                outs = self.model(self.normalize(x_ness))
                loss = self.lossfn(outs, y)

                if self.targeted:
                    loss = -loss

                # Compute gradient
                loss.backward()

                if delta.grad is None:
                    continue

                grad += delta.grad

            # Average gradient over scaled copies
            grad /= self.m

            # Apply momentum term
            g = self.decay * grad + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

        return x + delta


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(SINIFGSM, {'eps': 8 / 255, 'steps': 10, 'm': 3})
