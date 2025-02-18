from typing import Callable

import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class DeCoWA(Attack):
    """The DeCoWA (Deformation-Constrained Warping Attack) attack.

    > From the paper: [Boosting Adversarial Transferability across Model Genus by
    Deformation-Constrained Warping](https://arxiv.org/abs/2402.03951).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        mesh_width: Width of the control points. Defaults to 3.
        mesh_height: Height of the control points. Defaults to 3.
        rho: Regularization parameter for deformation. Defaults to 0.01.
        num_warping: Number of warping transformation samples. Defaults to 20.
        noise_scale: Scale of the random noise. Defaults to 2.
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
        mesh_width: int = 3,
        mesh_height: int = 3,
        rho: float = 0.01,
        num_warping: int = 20,
        noise_scale: int = 2,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.rho = rho
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform DeCoWA on a batch of images.

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

        for _ in range(self.steps):
            g = torch.zeros_like(x)

            # Perform warping
            for _ in range(self.num_warping):
                # Apply warping to perturbation
                noise_map = torch.rand([self.mesh_height - 2, self.mesh_width - 2, 2])
                noise_map_hat = (noise_map - 0.5) * self.noise_scale

                # Iterate for a round only
                for _ in range(1):
                    noise_map_hat.requires_grad_()

                    vwt_x = self._vwt(x + delta, noise_map_hat)
                    outs = self.model(self.normalize(vwt_x))
                    loss = self.lossfn(outs, y)

                    if self.targeted:
                        loss = -loss

                    loss.backward()

                    if noise_map_hat.grad is None:
                        continue

                    noise_map_hat.detach_()
                    noise_map_hat -= self.rho * noise_map_hat.grad

                vwt_x = self._vwt(x + delta, noise_map_hat)

                # Compute loss
                outs = self.model(self.normalize(vwt_x))
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

            # Average gradient over warping
            g /= self.num_warping

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

    def _vwt(self, x: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        """
        Performs Vanilla Warping Transformation (VWT) on the input images using the
        noise map, i.e., computes a grid of control points and applies TPS to warp the
        image according to these points.

        Args:
            x: The input tensor of shape (N, C, H, W).
            noise_map: The noise map tensor of shape (H, W, 2) used to warp the grid.

        Returns:
            The transformed tensor of the same shape as the input tensor `x`.
        """

        n, _, w, h = x.size()

        xx = self._grid_points_2d(self.mesh_width, self.mesh_height, self.device)
        yy = self._noisy_grid(self.mesh_width, self.mesh_height, noise_map, self.device)

        tpbs = TPS(size=(h, w), device=self.device)
        warped_grid_b = tpbs(xx[None, ...], yy[None, ...])
        warped_grid_b = warped_grid_b.repeat(n, 1, 1, 1)

        vwt_x = torch.grid_sampler_2d(
            input=x,
            grid=warped_grid_b,
            interpolation_mode=0,
            padding_mode=0,
            align_corners=False,
        )

        return vwt_x

    def _grid_points_2d(
        self, width: int, height: int, device: torch.device
    ) -> torch.Tensor:
        """Helper function to generate 2d grid points for the TPS transformation.

        Creates a regular grid of points in the range [-1, 1] x [-1, 1] with `width`
        points in the x-direction and `height` points in the y-direction. The grid is
        then reshaped to a 2D tensor of shape (width * height, 2).

        Args:
            width: The number of points in the x-direction.
            height: The number of points in the y-direction. device: The device.

        Returns:
            The grid of points of shape (width * height, 2).
        """

        x = torch.linspace(-1, 1, width, device=device)
        y = torch.linspace(-1, 1, height, device=device)

        grid_x, grid_y = torch.meshgrid(x, y)
        grid = torch.stack((grid_y, grid_x), dim=-1)

        return grid.view(-1, 2)

    def _noisy_grid(
        self, width: int, height: int, noise_map: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Creates a perturbed version of the grid by adding random noise.

        Args:
            width: The number of points in the x-direction.
            height: The number of points in the y-direction.
            noise_map: The noise map tensor of shape (H, W, 2) used to warp the grid.
            device: The device.

        Returns:
            The grid of points of shape (width * height, 2).
        """

        grid = self._grid_points_2d(width, height, device)
        mod = torch.zeros([height, width, 2], device=device)
        mod[1 : height - 1, 1 : width - 1, :] = noise_map
        return grid + mod.reshape(-1, 2)


def k_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the kernel matrix, which measures the deformation energy."""
    eps = 1e-9
    d2 = torch.pow(x[:, :, None, :] - y[:, None, :, :], 2).sum(-1)
    k_mat = d2 * torch.log(d2 + eps)
    return k_mat


def p_matrix(x: torch.Tensor) -> torch.Tensor:
    """Constructs the P matrix, which includes affine transformation components."""
    n, k = x.shape[:2]
    p_mat = torch.ones(n, k, 3, device=x.device)
    p_mat[:, :, 1:] = x
    return p_mat


class TPSCoeffs(nn.Module):
    """Computes the coefficients for the TPS transformation based on control points."""

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, k = x.shape[:2]
        z_mat = torch.zeros(1, k + 3, 2, device=x.device)
        p_mat = torch.ones(n, k, 3, device=x.device)
        l_mat = torch.zeros(n, k + 3, k + 3, device=x.device)
        k_mat = k_matrix(x, x)

        p_mat[:, :, 1:] = x
        z_mat[:, :k, :] = y
        l_mat[:, :k, :k] = k_mat
        l_mat[:, :k, k:] = p_mat
        l_mat[:, k:, :k] = p_mat.permute(0, 2, 1)

        q_mat = torch.linalg.solve(l_mat, z_mat)
        return q_mat[:, :k], q_mat[:, k:]


class TPS(nn.Module):
    """Thin Plate Spline transformation.

    Applies the TPS transformation to the image grid, warping it according to the
    computed coefficients.

    Args:
        size: The size of the grid. Defaults to (256, 256).
        device: Device to use for tensors. Defaults to cuda if available.
    """

    def __init__(self, size: tuple = (256, 256), device: torch.device | None = None):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device

        self.tps = TPSCoeffs()

        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w, device=device)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h, device=device)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h, w = self.size
        w_mat, a_mat = self.tps(x, y)
        u_mat = k_matrix(self.grid, x)
        p_mat = p_matrix(self.grid)
        grid: torch.Tensor = p_mat @ a_mat + u_mat @ w_mat
        return grid.view(-1, h, w, 2)


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=DeCoWA,
        attack_args={'eps': 16 / 255, 'steps': 10},
        model_name='resnet50',
        victim_model_names=['resnet18', 'vit_b_16'],
    )
