from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class SSA(Attack):
    """The SSA - Spectrum Simulation (S^2_I-FGSM) attack.

    > From the paper: [Frequency Domain Model Augmentation for Adversarial
    Attack](https://arxiv.org/abs/2207.05382).

    N.B.: This is implemented with momentum applied, i.e., on top of MI-FGSM.

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
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
        num_spectrum: int = 20,
        rho: float = 0.5,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.num_spectrum = num_spectrum
        self.rho = rho
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform SSA on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform SSA
        for _ in range(self.steps):
            g = torch.zeros_like(x)

            for _ in range(self.num_spectrum):
                # Frequency transformation (dct + idct)
                x_adv = self.transform(x + delta)

                # Compute loss
                outs = self.model(self.normalize(x_adv))
                loss = self.lossfn(outs, y)

                if self.targeted:
                    loss = -loss

                # Compute gradient
                loss.backward()

                if delta.grad is None:
                    continue

                # Accumulate gradient
                g += delta.grad

            if delta.grad is None:
                continue

            # Average gradient over num_spectrum
            g /= self.num_spectrum

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

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]

        # H and W must be a multiple of 8
        gauss = torch.randn(b, 3, 224, 224, device=x.device) * self.eps

        x_dct = self._dct_2d(x + gauss)
        mask = torch.rand_like(x) * 2 * self.rho + 1 - self.rho
        x_idct = self._idct_2d(x_dct * mask)

        return x_idct

    def _dct(self, x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Args:
            x: the input signal
            norm: the normalization, None or 'ortho'

        Returns:
            The DCT-II of the signal over the last dimension
        """

        x_shape = x.shape
        n = x_shape[-1]
        x = x.contiguous().view(-1, n)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        mat_v_c = torch.fft.fft(v)

        k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * n)
        mat_w_r = torch.cos(k)
        mat_w_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        mat_v = mat_v_c.real * mat_w_r - mat_v_c.imag * mat_w_i
        if norm == 'ortho':
            mat_v[:, 0] /= np.sqrt(n) * 2
            mat_v[:, 1:] /= np.sqrt(n / 2) * 2

        mat_v = 2 * mat_v.view(*x_shape)

        return mat_v  # type: ignore[no-any-return]

    def _idct(self, mat_x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Args:
            mat_x: the input signal
            norm: the normalization, None or 'ortho'

        Returns:
            The inverse DCT-II of the signal over the last dimension
        """

        x_shape = mat_x.shape
        n = x_shape[-1]

        mat_x_v = mat_x.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            mat_x_v[:, 0] *= np.sqrt(n) * 2
            mat_x_v[:, 1:] *= np.sqrt(n / 2) * 2

        k = (
            torch.arange(x_shape[-1], dtype=mat_x.dtype, device=mat_x.device)[None, :]
            * np.pi
            / (2 * n)
        )
        mat_w_r = torch.cos(k)
        mat_w_i = torch.sin(k)

        mat_v_t_r = mat_x_v
        mat_v_t_i = torch.cat([mat_x_v[:, :1] * 0, -mat_x_v.flip([1])[:, :-1]], dim=1)

        mat_v_r = mat_v_t_r * mat_w_r - mat_v_t_i * mat_w_i
        mat_v_i = mat_v_t_r * mat_w_i + mat_v_t_i * mat_w_r

        mat_v = torch.cat([mat_v_r.unsqueeze(2), mat_v_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=mat_v[:, :, 0], imag=mat_v[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, : n - (n // 2)]
        x[:, 1::2] += v.flip([1])[:, : n // 2]

        return x.view(*x_shape).real  # type: ignore[no-any-return]

    def _dct_2d(self, x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        """
        2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Args:
            x: the input signal
            norm: the normalization, None or 'ortho'

        Returns:
            The DCT-II of the signal over the last 2 dimensions
        """

        mat_x1 = self._dct(x, norm=norm)
        mat_x2 = self._dct(mat_x1.transpose(-1, -2), norm=norm)
        return mat_x2.transpose(-1, -2)

    def _idct_2d(self, mat_x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        """
        The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct_2d(dct_2d(x)) == x
        (This code is copied from https://github.com/yuyang-long/SSA/blob/master/dct.py)

        Args:
            mat_x: the input signal
            norm: the normalization, None or 'ortho'

        Returns:
            The DCT-II of the signal over the last 2 dimensions
        """

        x1 = self._idct(mat_x, norm=norm)
        x2 = self._idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=SSA,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
