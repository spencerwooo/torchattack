from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class DIFGSM(Attack):
    """The DI-FGSM (Diverse-input Iterative FGSM) attack.

    > From the paper: [Improving Transferability of Adversarial Examples with Input
    Diversity](https://arxiv.org/abs/1803.06978).

    Note:
        Key parameters include `resize_rate` and `diversity_prob`, which defines the
        scale size of the resized image and the probability of applying input
        diversity. The default values are set to 0.9 and 1.0 respectively (implying
        that input diversity is always applied).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        resize_rate: The resize rate. Defaults to 0.9.
        diversity_prob: Applying input diversity with probability. Defaults to 1.0.
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
        resize_rate: float = 0.9,
        diversity_prob: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform DI-FGSM on a batch of images.

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

        # Perform DI-FGSM
        for _ in range(self.steps):
            # Apply input diversity to intermediate images
            x_adv = input_diversity(x + delta, self.resize_rate, self.diversity_prob)

            # Compute loss
            outs = self.model(self.normalize(x_adv))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Apply momentum term
            g = self.decay * g + delta.grad / torch.mean(
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


def input_diversity(
    x: torch.Tensor, resize_rate: float = 0.9, diversity_prob: float = 0.5
) -> torch.Tensor:
    """Apply input diversity to a batch of images.

    Note:
        Adapted from the TensorFlow implementation (cihangxie/DI-2-FGSM):
        https://github.com/cihangxie/DI-2-FGSM/blob/10ffd9b9e94585b6a3b9d6858a9a929dc488fc02/attack.py#L153-L164

    Args:
        x: A batch of images. Shape: (N, C, H, W).
        resize_rate: The resize rate. Defaults to 0.9.
        diversity_prob: Applying input diversity with probability. Defaults to 0.5.

    Returns:
        The diversified batch of images. Shape: (N, C, H, W).
    """

    if torch.rand(1) > diversity_prob:
        return x

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = f.interpolate(x, size=[rnd, rnd], mode='nearest')

    h_rem = img_resize - rnd
    w_rem = img_resize - rnd

    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    pad = [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()]
    padded = f.pad(rescaled, pad=pad, mode='constant', value=0)

    return padded


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    args = {'eps': 16 / 255, 'steps': 10, 'resize_rate': 0.9, 'diversity_prob': 1.0}
    run_attack(DIFGSM, attack_args=args)
