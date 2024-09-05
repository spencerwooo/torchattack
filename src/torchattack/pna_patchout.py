import importlib.util
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.base import Attack


class PNAPatchOut(Attack):
    """PNA-PatchOut attack for ViTs (Pay no attention & PatchOut).

    From the paper: 'Towards Transferable Adversarial Attacks on Vision Transformers'
    https://arxiv.org/abs/2109.04176

    Args:
        model: The model to attack.
        model_name: The name of the model.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Momentum decay factor. Defaults to 1.0.
        pna_skip: Whether to apply PNA. Defaults to True.
        pna_patchout: Whether to apply PatchOut perturbation mask. Defaults to True.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    # fmt: off
    _supported_vit_cfg = {
        'vit_base_patch16_224': [
            f'blocks.{i}.attn.attn_drop' for i in range(12)
        ],
        'deit_base_distilled_patch16_224': [
            f'blocks.{i}.attn.attn_drop' for i in range(12)
        ],
        'pit_b_224': [
            f'transformers.{tid}.blocks.{i}.attn.attn_drop'
            for tid, bid in enumerate([3, 6, 4])
            for i in range(bid)
        ],
        'cait_s24_224': [
            # Regular blocks
            f'blocks.{i}.attn.attn_drop' for i in range(24)
        ] + [
            # Token-only block
            f'blocks_token_only.{i}.attn.attn_drop' for i in range(2)
        ],
        'visformer_small': [
            # Stage 2 blocks
            f'stage2.{i}.attn.attn_drop' for i in range(4)
        ] + [
            # Stage 3 blocks
            f'stage3.{i}.attn.attn_drop' for i in range(4)
        ],
    }
    # fmt: on

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        pna_skip: bool = True,
        pna_patchout: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        # Check if timm is installed
        importlib.util.find_spec('timm')

        super().__init__(normalize, device)

        self.model = model
        self.model_name = model_name
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.pna_skip = pna_skip
        self.pna_patchout = pna_patchout
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        if self.pna_skip:
            self._register_vit_model_hook()

        # Set default image size and number of patches for PatchOut
        self.image_size = 224
        self.crop_len = 16
        self.max_num_patches = int((self.image_size / self.crop_len) ** 2)
        if self.pna_patchout:
            self.sample_num_patches = 130
        else:
            self.sample_num_patches = self.max_num_patches

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(x)
        delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        for i in range(self.steps):
            patched_out_delta = self._apply_patch_out(delta, seed=i)
            outs = self.model(self.normalize(x + patched_out_delta))
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

    def _register_vit_model_hook(self):
        def attn_drop_mask_grad(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:],)

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        assert self.model_name in self._supported_vit_cfg

        # Register backward hook for layers specified in _supported_vit_cfg
        for layer in self._supported_vit_cfg[self.model_name]:
            module = rgetattr(self.model, layer)
            module.register_backward_hook(drop_hook_func)

    def _apply_patch_out(self, delta: torch.Tensor, seed: int) -> torch.Tensor:
        delta_mask = torch.zeros_like(delta)
        grid_num_axis = int(self.image_size / self.crop_len)

        # Randomly sample patches (unrepeatable)
        torch.manual_seed(seed)
        ids = torch.randperm(self.max_num_patches)[: self.sample_num_patches]

        # Repeatable sampling
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        for r, c in zip(rows, cols, strict=True):
            delta_mask[
                ...,
                r * self.crop_len : (r + 1) * self.crop_len,
                c * self.crop_len : (c + 1) * self.crop_len,
            ] = 1

        # Apply mask to delta
        return delta_mask * delta


if __name__ == '__main__':
    from torchattack.eval import run_attack

    run_attack(
        PNAPatchOut,
        attack_cfg={'model_name': 'vit_base_patch16_224'},
        model_name='vit_base_patch16_224',
        from_timm=True,
    )
