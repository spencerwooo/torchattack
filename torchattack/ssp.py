from typing import Callable

import torch
import torch.nn as nn
import torchvision as tv

from torchattack._attack import Attack
from torchattack.attack_model import AttackModel


class PerceptualCriteria(nn.Module):
    def __init__(self, ssp_layer: int) -> None:
        super().__init__()
        self.ssp_layer = ssp_layer

        # Use pretrained VGG16 for perceptual loss
        vgg16 = tv.models.vgg16(weights='DEFAULT')

        # Initialize perceptual model and loss function
        self.perceptual_model = nn.Sequential(*list(vgg16.features))[:ssp_layer]
        self.perceptual_model.eval()
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, xadv: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.perceptual_model(x), self.perceptual_model(xadv))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(ssp_layer={self.ssp_layer})'


class SSP(Attack):
    """The Self-supervised (SSP) attack.

    From the paper 'A Self-supervised Approach for Adversarial Robustness'
    https://arxiv.org/abs/2006.04924

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 100.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        ssp_layer: The VGG layer to use for the perceptual loss. Defaults to 16.
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
        steps: int = 100,
        alpha: float | None = None,
        ssp_layer: int = 16,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.ssp_layer = ssp_layer
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

        self.perceptual_criteria = PerceptualCriteria(ssp_layer).to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform SSP on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels, not required. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        delta = torch.randn_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        for _ in range(self.steps):
            xadv = x + delta
            loss = self.perceptual_criteria(self.normalize(x), self.normalize(xadv))
            loss.backward()

            if delta.grad is None:
                continue

            # Update delta
            g = delta.grad.data.sign()

            delta.data = delta.data + self.alpha * g
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta


if __name__ == '__main__':
    from torchattack.eval import run_attack

    run_attack(SSP, attack_args={'eps': 16 / 255, 'ssp_layer': 16})
