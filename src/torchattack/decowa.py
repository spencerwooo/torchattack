from typing import Callable

import torch
import torch.nn as nn

from torchattack.base import Attack


class DeCoWA(Attack):
    """The DeCoWA (Deformation-Constrained Warping Attack) attack.

    From the paper 'Boosting Adversarial Transferability across Model Genus by
    Deformation-Constrained Warping',
    https://arxiv.org/abs/2402.03951

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
        model: nn.Module,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
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
        super().__init__(normalize, device)

        self.model = model
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

    def forward():
        pass
