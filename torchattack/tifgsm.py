from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class TIFGSM(Attack):
    """The TI-FGSM (Translation-invariant Iterative FGSM) attack.

    > From the paper: [Evading Defenses to Transferable Adversarial Examples by
    Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884).

    Note:
        Key parameters include `kernel_len` and `n_sig`, which defines the size and
        the radius of the gaussian kernel. The default values are set to 15 and 3
        respectively, which are best according to the paper.

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        kern_len: Length of the kernel (should be an odd number). Defaults to 15.
        n_sig: Radius of the gaussian kernel. Defaults to 3.
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
        kern_len: int = 15,
        n_sig: int = 3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.kern_len = kern_len
        self.n_sig = n_sig
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform TI-FGSM on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        g = torch.zeros_like(x)
        delta = torch.zeros_like(x, requires_grad=True)

        # Get kernel
        kernel = self.get_kernel()

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform TI-FGSM
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

            # Apply kernel to gradient
            g = f.conv2d(delta.grad, kernel, stride=1, padding='same', groups=3)

            # Apply momentum term
            g = self.decay * g + g / torch.mean(
                torch.abs(g), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta

    def get_kernel(self) -> torch.Tensor:
        kernel = self.gkern(self.kern_len, self.n_sig).astype(np.float32)

        kernel = np.expand_dims(kernel, axis=0)  # (W, H) -> (1, W, H)
        kernel = np.repeat(kernel, 3, axis=0)  # -> (C, W, H)
        kernel = np.expand_dims(kernel, axis=1)  # -> (C, 1, W, H)
        return torch.from_numpy(kernel).to(self.device)

    @staticmethod
    def gkern(kern_len: int = 15, n_sig: int = 3) -> np.ndarray:
        """Return a 2D Gaussian kernel array."""

        import scipy.stats as st

        interval = (2 * n_sig + 1.0) / kern_len
        x = np.linspace(-n_sig - interval / 2.0, n_sig + interval / 2.0, kern_len + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        return np.array(kernel_raw / kernel_raw.sum(), dtype=np.float32)


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(TIFGSM, {'eps': 8 / 255, 'steps': 10, 'kern_len': 15, 'n_sig': 3})
