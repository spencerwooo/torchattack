from typing import Callable

import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class MIG(Attack):
    """The MIG (Momentum Integrated Gradients) attack.

    > From the paper: [Transferable Adversarial Attack for Both Vision Transformers and
    Convolutional Networks via Momentum Integrated Gradients](https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        s_factor: Number of scaled interpolation iterations, $T$. Defaults to 25.
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
        s_factor: int = 25,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.s_factor = s_factor
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def _get_scaled_samples(self, x: torch.Tensor) -> torch.Tensor:
        xb = torch.zeros_like(x)
        xss = [xb + (i + 1) / self.s_factor * (x - xb) for i in range(self.s_factor)]
        return torch.cat(xss, dim=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform MIG on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        g = torch.zeros_like(x)
        delta = torch.zeros_like(x, requires_grad=True)
        # delta.data.uniform_(-self.eps, self.eps)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        xbase = torch.zeros_like(x)

        # Perform MIG
        for _ in range(self.steps):
            # Compute loss
            scaled_samples = self._get_scaled_samples(x + delta)
            logits = self.model(self.normalize(scaled_samples))

            # Softmax over logits
            probs = nn.functional.softmax(logits, dim=1)

            # Get loss
            loss = torch.mean(probs.gather(1, y.repeat(self.s_factor).view(-1, 1)))

            if self.targeted:
                loss = -loss

            # Compute gradient over backprop
            loss.backward()

            if delta.grad is None:
                continue

            # Integrated gradient
            igrad = (x + delta - xbase) * delta.grad / self.s_factor

            # Apply momentum term
            g = self.decay * g + igrad / torch.mean(
                torch.abs(igrad), dim=(1, 2, 3), keepdim=True
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
    from torchattack.evaluate import run_attack

    run_attack(
        attack=MIG,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='timm/vit_base_patch16_224',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
        save_adv_batch=1,
    )
