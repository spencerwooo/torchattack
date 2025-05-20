from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack(category='GRADIENT_VIT')
class ATT(Attack):
    """The ATT (Adaptive Token Tuning) attack for ViTs.

    > From the paper: [Boosting the Transferability of Adversarial Attack on Vision
    Transformer with Adaptive Token Tuning](https://openreview.net/forum?id=sNz7tptCH6).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        hook_cfg: Config used for applying hooks to the model. Supported values:
            `vit_base_patch16_224`, `pit_b_224`, `cait_s24_224`, `visformer_small`.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        lambd: Lambda value for the gradient factor. Defaults to 0.01.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        hook_cfg: str = '',
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        lambd: float = 0.01,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        if hook_cfg:
            # Explicit config name takes precedence over inferred model.model_name
            self.hook_cfg = hook_cfg
        elif isinstance(model, AttackModel):
            # If model is initialized via `torchattack.AttackModel`, the model_name
            # is automatically attached to the model during instantiation.
            self.hook_cfg = model.model_name

        # Surrogate ViT for VDC must be `timm` models or models that have the same
        # structure and same implementation/definition as `timm` models. Note that we
        # delay parent init to avoid overriding the model's `model_name` attribute
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.lambd = lambd
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        self.img_size = 224
        self.crop_len = 16

        self.mid_feats = torch.tensor([])
        self.mid_grads = torch.tensor([])
        self.patch_index = self._patch_index(self.img_size, self.crop_len).to(device)
        self.gamma = torch.tensor(0.5)

        if self.hook_cfg not in [
            'vit_base_patch16_224',
            'pit_b_224',
            'cait_s24_224',
            'visformer_small',
        ]:
            from warnings import warn

            warn(
                f'Hook config specified (`{self.hook_cfg}`) is not supported. '
                'Falling back to default (`vit_base_patch16_224`). '
                'This MAY NOT be intended.',
                stacklevel=2,
            )
            self.hook_cfg = 'vit_base_patch16_224'

        # Initialize layer and patch parameters
        self._init_params()

        # Register hooks for the model
        self._register_vit_model_hook()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform ATT on a batch of images.

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

        # Forward and backward pass with dummy input to init gradient + features
        output = self.model(self.normalize(x))
        output.backward(torch.ones_like(output))

        gf = self._get_gf()

        # Normalize and expand gradient features to match the original image size
        gf_patchs_t = self._norm_patchs(
            gf, self.patch_index, self.crop_len, self.scale, self.offset
        )

        # Initialize the starting threshold for gradient features patches
        gf_patchs_start = torch.ones_like(gf_patchs_t) * 0.99

        # Calculate the offset for gradient features patches for each step
        gf_offset = (gf_patchs_start - gf_patchs_t) / self.steps

        for i in range(self.steps):
            self._reset_vars()

            # Init random patch
            random_patch = torch.rand_like(x)

            # Calculate the threshold for gradient features patches for the current step
            gf_patchs_threshold = gf_patchs_start - gf_offset * (i + 1)

            # Create a mask for the gradient features patches based on the threshold
            gf_patchs = torch.where(random_patch > gf_patchs_threshold, 0.0, 1.0)

            # Apply gradient feature patches to the perturbation
            outs = self.model(self.normalize(x + delta * gf_patchs))
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

    def _get_gf(self) -> torch.Tensor:
        if self.hook_cfg == 'vit_base_patch16_224':
            gf = (self.mid_feats[0][1:] * self.mid_grads[0][1:]).sum(-1)
            gf = self._resize(gf.reshape(1, 14, 14), self.img_size)
        elif self.hook_cfg == 'pit_b_224':
            gf = (self.mid_feats[0][1:] * self.mid_grads[0][1:]).sum(-1)
            gf = self._resize(gf.reshape(1, 8, 8), self.img_size)
        elif self.hook_cfg == 'cait_s24_224':
            gf = (self.mid_feats[0] * self.mid_grads).sum(-1)
            gf = self._resize(gf.reshape(1, 14, 14), self.img_size)
        elif self.hook_cfg == 'visformer_small':
            gf = (self.mid_feats[0] * self.mid_grads).sum(0)
            gf = self._resize(gf.unsqueeze(0), self.img_size)
        else:
            raise ValueError(f'Unsupported hook config: {self.hook_cfg}')
        return gf

    def _init_params(self) -> None:
        self._reset_vars()

        if self.hook_cfg == 'vit_base_patch16_224':
            self.truncate_layers = self._tr_01_pc(10, 12)
            self.weaken_factor = [0.45, 0.7, 0.65]
            self.scale = 0.4
            self.offset = 0.4
        elif self.hook_cfg == 'pit_b_224':
            self.truncate_layers = self._tr_01_pc(9, 13)
            self.weaken_factor = [0.25, 0.6, 0.65]
            self.scale = 0.3
            self.offset = 0.45
        elif self.hook_cfg == 'cait_s24_224':
            self.truncate_layers = self._tr_01_pc(4, 25)
            self.weaken_factor = [0.3, 1.0, 0.6]
            self.scale = 0.35
            self.offset = 0.4
        elif self.hook_cfg == 'visformer_small':
            self.truncate_layers = self._tr_01_pc(8, 8)
            self.weaken_factor = [0.4, 0.8, 0.3]
            self.scale = 0.15
            self.offset = 0.25
        else:
            raise ValueError(f'Unsupported hook config: {self.hook_cfg}')

    def _reset_vars(self) -> None:
        self.var_a = torch.tensor(0)
        self.var_qkv = torch.tensor(0)
        self.var_mlp = torch.tensor(0)

        if self.hook_cfg == 'vit_base_patch16_224':
            self.back_attn = 11
        elif self.hook_cfg == 'pit_b_224':
            self.back_attn = 12
        elif self.hook_cfg == 'cait_s24_224':
            self.back_attn = 24
        elif self.hook_cfg == 'visformer_small':
            self.back_attn = 7
        else:
            raise ValueError(f'Unsupported hook config: {self.hook_cfg}')

    @staticmethod
    def _resize(x: torch.Tensor, img_size: int) -> torch.Tensor:
        """Simplified version of `torchvision.transforms.Resize`."""

        need_squeeze = False
        # make image NCHW
        if x.ndim < 4:
            x = x.unsqueeze(dim=0)
            need_squeeze = True
        x = torch.nn.functional.interpolate(
            x,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False,
        )
        return x.squeeze(0) if need_squeeze else x

    @staticmethod
    def _tr_01_pc(num: int, len: int) -> torch.Tensor:
        """Create a tensor with 1s at the first `num` indices and 0s elsewhere."""

        return torch.cat((torch.ones(num), torch.zeros(len - num)))

    @staticmethod
    def _patch_index(img_size: int, crop_len: int) -> torch.Tensor:
        """Create indices for patches in an image."""

        # Calculate number of patches in each dimension
        num_patches = int(np.floor((img_size - crop_len) / crop_len) + 1)

        # Create base indices for a single patch
        row_indices = np.arange(crop_len)[:, None] * img_size
        col_indices = np.arange(crop_len)
        patch_base = row_indices + col_indices

        # Create offset matrices for patches
        row_offsets = (np.arange(num_patches) * crop_len * img_size)[:, None]
        col_offsets = np.arange(num_patches) * crop_len

        # Combine offsets to get all patch positions
        patch_positions = row_offsets.reshape(-1, 1) + col_offsets.reshape(1, -1)
        patch_positions = patch_positions.reshape(-1, 1)

        # Add base indices to each patch position
        indices = patch_positions + patch_base.ravel()

        # Convert to correct shape and return as tensor
        return torch.LongTensor(indices).unsqueeze(0)

    @staticmethod
    def _norm_patchs(
        gf: torch.Tensor,
        patch_index: torch.Tensor,
        patch_size: int,
        scale: float,
        offset: float,
    ) -> torch.Tensor:
        """Normalize and expand patch values to match the original image size."""

        # Extract patch values using indices
        patch_values = torch.take(gf, patch_index)

        # Calculate mean for each patch
        patch_means = torch.mean(patch_values, dim=-1, keepdim=True)

        # Min-max normalization of patch means
        min_values = patch_means.min(dim=-1, keepdim=True)[0]
        max_values = patch_means.max(dim=-1, keepdim=True)[0]
        normalized_values = (patch_means - min_values) / (max_values - min_values)

        # Apply scaling and offset
        scaled_values = scale * normalized_values + offset

        # Expand normalized values to match patch size
        expanded_values = scaled_values.repeat_interleave(patch_size**2, dim=-1)

        # Update gradient features with normalized values
        gf = gf.put_(patch_index, expanded_values)

        return gf

    def _register_vit_model_hook(self) -> None:
        def attn_att(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            mask = (
                torch.ones_like(grad_in[0])
                * self.truncate_layers[self.back_attn]
                * self.weaken_factor[0]
            )
            out_grad = mask * grad_in[0][:]
            if self.var_a.item() == 0:
                gpf = self.gamma
            else:
                gpf = 1 - torch.sqrt(torch.var(out_grad) / self.var_a)
                gpf = (self.gamma + self.lambd * gpf).clamp(0, 1)
            if self.hook_cfg in [
                'vit_base_patch16_224',
                'visformer_small',
                'pit_b_224',
            ]:
                b, c, h, w = grad_in[0].shape
                out_grad_reshaped = out_grad.reshape(b, c, h * w)
                max_all = out_grad_reshaped[0].max(dim=1)[1]
                min_all = out_grad_reshaped[0].min(dim=1)[1]
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all_h = min_all // h
                min_all_w = min_all % h

                out_grad[:, range(c), max_all_h, :] *= gpf
                out_grad[:, range(c), :, max_all_w] *= gpf
                out_grad[:, range(c), min_all_h, :] *= gpf
                out_grad[:, range(c), :, min_all_w] *= gpf

            if self.hook_cfg in ['cait_s24_224']:
                b, h, w, c = grad_in[0].shape
                out_grad_reshaped = out_grad.reshape(b, h * w, c)
                max_all = out_grad_reshaped[0, :, :].max(dim=0)[1]
                min_all = out_grad_reshaped[0, :, :].min(dim=0)[1]
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all_h = min_all // h
                min_all_w = min_all % h

                out_grad[:, max_all_h, :, range(c)] *= gpf
                out_grad[:, :, max_all_w, range(c)] *= gpf
                out_grad[:, min_all_h, :, range(c)] *= gpf
                out_grad[:, :, min_all_w, range(c)] *= gpf

            self.var_a = torch.var(out_grad)

            self.back_attn -= 1
            return (out_grad,)

        def attn_cait_att(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            mask = (
                torch.ones_like(grad_in[0])
                * self.truncate_layers[self.back_attn]
                * self.weaken_factor[0]
            )

            out_grad = mask * grad_in[0][:]
            if self.var_a.item() == 0:
                gpf = self.gamma
            else:
                gpf = 1 - torch.sqrt(torch.var(out_grad) / self.var_a)
                gpf = (self.gamma + self.lambd * gpf).clamp(0, 1)
            b, h, w, c = grad_in[0].shape
            out_grad_reshaped = out_grad.reshape(b, h * w, c)
            max_all = out_grad_reshaped[0, :, :].max(dim=0)[1]
            min_all = out_grad_reshaped[0, :, :].min(dim=0)[1]
            max_all_h = max_all // h
            max_all_w = max_all % h
            min_all_h = min_all // h
            min_all_w = min_all % h

            out_grad[:, max_all_h, :, range(c)] *= gpf
            out_grad[:, :, max_all_w, range(c)] *= gpf
            out_grad[:, min_all_h, :, range(c)] *= gpf
            out_grad[:, :, min_all_w, range(c)] *= gpf

            self.var_a = torch.var(out_grad)
            self.back_attn -= 1
            return (out_grad,)

        def q_att(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            if self.var_qkv.item() == 0:
                gpf = self.gamma
            else:
                gpf = 1 - torch.sqrt(torch.var(out_grad) / self.var_qkv)
                gpf = (self.gamma + self.lambd * gpf).clamp(0, 1)
            out_grad[:] *= gpf
            self.var_qkv = torch.var(out_grad)
            return (out_grad, grad_in[1], grad_in[2])

        def v_att(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            is_dim_extra = False
            if len(grad_in[0].shape) == 2:
                is_dim_extra = True
                grad_in = (grad_in[0].unsqueeze(0),) + grad_in[1:]

            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            if self.var_qkv.item() == 0:
                gpf = self.gamma
            else:
                gpf = 1 - torch.sqrt(torch.var(out_grad) / self.var_qkv)
                gpf = (self.gamma + self.lambd * gpf).clamp(0, 1)

            if self.hook_cfg in ['visformer_small']:
                b, c, h, w = grad_in[0].shape
                out_grad_reshaped = out_grad.reshape(b, c, h * w)
                max_all = out_grad_reshaped[0].max(dim=1)[1]
                min_all = out_grad_reshaped[0].min(dim=1)[1]
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all_h = min_all // h
                min_all_w = min_all % h

                out_grad[:, range(c), max_all_h, max_all_w] *= gpf
                out_grad[:, range(c), min_all_h, min_all_w] *= gpf

            if self.hook_cfg in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_reshaped = out_grad[0]
                max_all = out_grad_reshaped.max(dim=0)[1]
                min_all = out_grad_reshaped.min(dim=0)[1]

                out_grad[:, max_all, range(c)] *= gpf
                out_grad[:, min_all, range(c)] *= gpf

            if is_dim_extra:
                out_grad = out_grad.squeeze(0)

            self.var_qkv = torch.var(out_grad)
            return (out_grad,) + tuple(grad_in[1:])

        def mlp_att(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            is_dim_extra = False
            if len(grad_in[0].shape) == 2:
                is_dim_extra = True
                grad_in = (grad_in[0].unsqueeze(0),) + grad_in[1:]

            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[2]
            out_grad = mask * grad_in[0][:]
            if self.var_mlp.item() == 0:
                gpf = self.gamma
            else:
                gpf = 1 - torch.sqrt(torch.var(out_grad) / self.var_mlp)
                gpf = (self.gamma + self.lambd * gpf).clamp(0, 1)

            if self.hook_cfg in ['visformer_small']:
                b, c, h, w = grad_in[0].shape
                out_grad_reshaped = out_grad.reshape(b, c, h * w)
                max_all = out_grad_reshaped[0].max(dim=1)[1]
                min_all = out_grad_reshaped[0].min(dim=1)[1]
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all_h = min_all // h
                min_all_w = min_all % h
                out_grad[:, range(c), max_all_h, max_all_w] *= gpf
                out_grad[:, range(c), min_all_h, min_all_w] *= gpf

            if self.hook_cfg in [
                'vit_base_patch16_224',
                'pit_b_224',
                'cait_s24_224',
            ]:
                c = grad_in[0].shape[2]
                out_grad_reshaped = out_grad[0]
                max_all = out_grad_reshaped.max(dim=0)[1]
                min_all = out_grad_reshaped.min(dim=0)[1]

                out_grad[:, max_all, range(c)] *= gpf
                out_grad[:, min_all, range(c)] *= gpf

            if is_dim_extra:
                out_grad = out_grad.squeeze(0)

            self.var_mlp = torch.var(out_grad)
            return (out_grad,) + grad_in[1:]

        def mid_feats_hook(m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
            self.mid_feats = o.clone()

        def mid_grads_hook(m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
            self.mid_grads = o[0].clone()

        # fmt: off
        feature_grad_hook_cfg = {
            'vit_base_patch16_224': 'blocks.10',
            'pit_b_224': 'transformers.2.blocks.2',
            'cait_s24_224': 'blocks.23',
            'visformer_small': 'stage3.2'
        }

        attention_hook_cfg = {
            'vit_base_patch16_224': [
                (attn_att, [f'blocks.{i}.attn.attn_drop' for i in range(12)]),
                (v_att, [f'blocks.{i}.attn.qkv' for i in range(12)]),
                (mlp_att, [f'blocks.{i}.mlp' for i in range(12)]),
            ],
            'pit_b_224': [
                (attn_att, [f'transformers.{tid}.blocks.{i}.attn.attn_drop' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
                (v_att, [f'transformers.{tid}.blocks.{i}.attn.qkv' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
                (mlp_att, [f'transformers.{tid}.blocks.{i}.mlp' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
            ],
            'cait_s24_224': [
                (attn_att, [f'blocks.{i}.attn.attn_drop' for i in range(24)]),
                (v_att, [f'blocks.{i}.attn.qkv' for i in range(24)]),
                (mlp_att, [f'blocks.{i}.mlp' for i in range(24)]),
                (attn_cait_att, [f'blocks_token_only.{i}.attn.attn_drop' for i in range(2)]),
                (q_att, [f'blocks_token_only.{i}.attn.q' for i in range(2)]),
                (v_att, [f'blocks_token_only.{i}.attn.k' for i in range(2)] + [f'blocks_token_only.{i}.attn.v' for i in range(2)]),
                (mlp_att, [f'blocks_token_only.{i}.mlp' for i in range(2)]),
            ],
            'visformer_small': [
                (attn_att, [f'stage2.{i}.attn.attn_drop' for i in range(4)] + [f'stage3.{i}.attn.attn_drop' for i in range(4)]),
                (v_att, [f'stage2.{i}.attn.qkv' for i in range(4)] + [f'stage3.{i}.attn.qkv' for i in range(4)]),
                (mlp_att, [f'stage2.{i}.mlp' for i in range(4)] + [f'stage3.{i}.mlp' for i in range(4)]),
            ],
        }
        # fmt: on

        assert feature_grad_hook_cfg.keys() == attention_hook_cfg.keys()
        assert self.hook_cfg in feature_grad_hook_cfg

        # Register feature and gradient hooks
        module = rgetattr(self.model, feature_grad_hook_cfg[self.hook_cfg])
        module.register_forward_hook(mid_feats_hook)
        if self.hook_cfg in ['vit_base_patch16_224', 'pit_b_224']:
            module.register_backward_hook(mid_grads_hook)
        else:
            module.register_forward_hook(mid_grads_hook)  # not sure why

        # Register attention hooks
        for hook_func, layers in attention_hook_cfg[self.hook_cfg]:
            for layer in layers:
                module = rgetattr(self.model, layer)
                module.register_backward_hook(hook_func)


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        ATT,
        model_name='timm/vit_base_patch16_224',
        victim_model_names=['timm/pit_b_224', 'timm/cait_s24_224'],
        batch_size=4,
    )
