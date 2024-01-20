from typing import Callable

import torch
import torch.nn as nn

from torchattack.base import Attack


class NIFGSM(Attack):
    """The NI-FGSM (Nesterov-accelerated Iterative FGSM) attack.

    Note:
        This attack does not apply the scale-invariant method. For the original attack
        proposed in the paper (SI-NI-FGSM), see `torchattack.sinifgsm.SINIFGSM`.

    From the paper 'Nesterov Accelerated Gradient and Scale Invariance for Adversarial
    Attacks' https://arxiv.org/abs/1908.06281
    """

    def __init__(
        self,
        model: nn.Module,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the NI-FGSM attack.

        Args:
            model: The model to attack.
            normalize: A transform to normalize images.
            eps: The maximum perturbation. Defaults to 8/255.
            steps: Number of steps. Defaults to 10.
            alpha: Step size, `eps / steps` if None. Defaults to None.
            decay: Decay factor for the momentum term. Defaults to 1.0.
            clip_min: Minimum value for clipping. Defaults to 0.0.
            clip_max: Maximum value for clipping. Defaults to 1.0.
            targeted: Targeted attack if True. Defaults to False.
            device: Device to use for tensors. Defaults to cuda if available.
        """

        super().__init__(device, normalize)

        self.model = model
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform NI-FGSM on a batch of images.

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

        # Perform NI-FGSM
        for _ in range(self.steps):
            # Nesterov gradient component
            nes = self.alpha * self.decay * g
            x_nes = x + delta + nes

            # Compute loss
            outs = self.model(self.normalize(x_nes))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Apply momentum term
            g = self.decay * delta.grad + delta.grad / torch.mean(
                torch.abs(delta.grad), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta


if __name__ == '__main__':
    from torchattack.utils import run_attack

    run_attack(NIFGSM, {'eps': 8 / 255, 'steps': 10})
