from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torchattack._attack import Attack
from torchattack._rgetattr import rgetattr
from torchattack.attack_model import AttackModel


class ATT(Attack):
    """The ATT (Adaptive Token Tuning) attack for ViTs.

    From the paper: [Boosting the Transferability of Adversarial Attack on Vision
    Transformer with Adaptive Token Tuning](https://openreview.net/forum?id=sNz7tptCH6).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        hook_cfg: Config used for applying hooks to the model. Supported configs are:
            * 'vit_base_patch16_224'
            * 'pit_b_224'
            * 'cait_s24_224'
            * 'visformer_small'
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Momentum decay factor. Defaults to 1.0.
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
        sample_num_batches: int = 130,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        # Surrogate ViT for VDC must be `timm` models or models that have the same
        # structure and same implementation/definition as `timm` models.
        super().__init__(model, normalize, device)

        if hook_cfg:
            # Explicit config name takes precedence over inferred model.model_name
            self.hook_cfg = hook_cfg
        elif hasattr(model, 'model_name'):
            # If model is initialized via `torchattack.AttackModel`, the model_name
            # is automatically attached to the model during instantiation.
            self.hook_cfg = model.model_name

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.sample_num_patches = sample_num_batches
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        # Set default image size and number of patches
        self.image_size = 224
        self.crop_len = 16
        self.max_num_patches = int((self.image_size / self.crop_len) ** 2)

        self.mid_feats = torch.tensor([])
        self.mid_grads = torch.tensor([])
        self.lambd = 0.01  # lambda
        self.patch_index = self._patch_index(self.image_size, self.crop_len).to(device)

        # Check if hook config is supported
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

        # Register hooks
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

        # Perform a single forward pass with clean sample and backward pass with
        # dummy data, to populate self.mid_feats and self.mid_grads
        outs = self.model(self.normalize(x))
        outs.backward(torch.ones_like(outs))
        gf = self._get_gf()

        gf_patchs_t = self._norm_patchs(
            gf, self.patch_index, self.crop_len, self.scale, self.offset
        )
        gf_patchs_start = torch.ones_like(gf_patchs_t) * 0.99
        gf_offset = (gf_patchs_start - gf_patchs_t) / self.steps

        for i in range(self.steps):
            random_patch = torch.rand_like(x)
            gf_patchs_threshold = gf_patchs_start - gf_offset * (i + 1)
            gf_patchs = torch.where(random_patch > gf_patchs_threshold, 0.0, 1.0)

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

    def _patch_index(self, image_size: int, crop_len: int) -> torch.Tensor:
        # Calculate number of patches in each dimension
        mat_dim = torch.floor(torch.tensor((image_size - crop_len) / crop_len) + 1)
        mat_dim = mat_dim.to(torch.int64)

        # Create base indices template for a single patch
        base_indices = torch.arange(crop_len, dtype=torch.int64)

        # Generate row and column offsets for each element within a patch
        row_offsets = (base_indices.view(-1, 1) * image_size).repeat(1, crop_len)
        col_offsets = base_indices.repeat(crop_len, 1)

        # Combine offsets to get indices for elements in a single patch
        patch_indices = (row_offsets + col_offsets).reshape(-1)

        # Generate offset grid for all patch positions
        stride_row_offsets = torch.arange(mat_dim, dtype=torch.int64).view(-1, 1)
        stride_row_offsets = stride_row_offsets * crop_len * image_size
        stride_row_offsets = stride_row_offsets.repeat(1, mat_dim)

        stride_col_offsets = torch.arange(mat_dim, dtype=torch.int64)
        stride_col_offsets = (stride_col_offsets * crop_len).repeat(mat_dim, 1)

        stride_offsets = (stride_row_offsets + stride_col_offsets).reshape(-1, 1)

        # Combine base patch indices with position offsets
        index = patch_indices.view(1, -1) + stride_offsets

        # Add batch dimension and move to device
        return index.unsqueeze(0).repeat(1, 1, 1)

    def _norm_patchs(
        self,
        gf: torch.Tensor,
        index: torch.Tensor,
        patch: int,
        scale: float,
        offset: float,
    ) -> torch.Tensor:
        patch_size = patch**2
        tmp = torch.take(gf, index)
        norm_tmp = torch.mean(tmp, dim=-1, keepdim=True)
        scale_norm = (
            scale
            * (
                (norm_tmp - norm_tmp.min(dim=-1, keepdim=True)[0])
                / (
                    norm_tmp.max(dim=-1, keepdim=True)[0]
                    - norm_tmp.min(dim=-1, keepdim=True)[0]
                )
            )
            + offset
        )
        tmp_bi = scale_norm.repeat_interleave(patch_size, dim=-1)
        gf = gf.put_(index, tmp_bi)
        return gf

    def _tr_01_pc(self, num: int, length: int) -> torch.Tensor:
        return torch.cat((torch.ones(num), torch.zeros(length - num)))

    def _get_gf(self) -> torch.Tensor:
        def _resize(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        def _vit_base_patch16_224_gf() -> torch.Tensor:
            return torch.sum(self.mid_feats[0][1:] * self.mid_grads[0][1:], dim=-1)

        def _pit_b_224_gf() -> torch.Tensor:
            return torch.sum(self.mid_feats[0][1:] * self.mid_grads[0][1:], dim=-1)

        def _cait_s24_224_gf() -> torch.Tensor:
            return torch.sum(self.mid_feats[0] * self.mid_grads, dim=-1)

        def _visformer_small_gf() -> torch.Tensor:
            return torch.sum(self.mid_feats[0] * self.mid_grads, dim=0)

        gf_cfg = {
            'vit_base_patch16_224': (_vit_base_patch16_224_gf, (1, 14, 14)),
            'pit_b_224': (_pit_b_224_gf, (1, 8, 8)),
            'cait_s24_224': (_cait_s24_224_gf, (1, 14, 14)),
            'visformer_small': (_visformer_small_gf, None),
        }
        assert self.hook_cfg in gf_cfg

        get_grad_feat_fn, reshape_size = gf_cfg[self.hook_cfg]

        gf = get_grad_feat_fn()
        gf = gf.reshape(*reshape_size) if reshape_size else gf.unsqueeze(0)
        return _resize(gf)

    def _register_vit_model_hook(self) -> None:
        self.var_a = torch.tensor(0)
        self.var_qkv = torch.tensor(0)
        self.var_mlp = torch.tensor(0)
        self.gamma = torch.tensor(0.5)

        @dataclass
        class HookParams:
            back_attn: int
            truncate_layers: torch.Tensor
            weaken_factor: list[float]
            scale: float
            offset: float

        hook_params_cfg = {
            'vit_base_patch16_224': HookParams(
                back_attn=11,
                truncate_layers=self._tr_01_pc(10, 12),
                weaken_factor=[0.45, 0.7, 0.65],
                scale=0.4,
                offset=0.4,
            ),
            'pit_b_224': HookParams(
                back_attn=12,
                truncate_layers=self._tr_01_pc(9, 13),
                weaken_factor=[0.25, 0.6, 0.65],
                scale=0.3,
                offset=0.45,
            ),
            'cait_s24_224': HookParams(
                back_attn=24,
                truncate_layers=self._tr_01_pc(4, 25),
                weaken_factor=[0.3, 1.0, 0.6],
                scale=0.35,
                offset=0.4,
            ),
            'visformer_small': HookParams(
                back_attn=7,
                truncate_layers=self._tr_01_pc(8, 8),
                weaken_factor=[0.4, 0.8, 0.3],
                scale=0.15,
                offset=0.25,
            ),
        }
        assert self.hook_cfg in hook_params_cfg

        hook_params = hook_params_cfg[self.hook_cfg]
        self.back_attn = hook_params.back_attn
        self.truncate_layers = hook_params.truncate_layers
        self.weaken_factor = hook_params.weaken_factor
        self.scale = hook_params.scale
        self.offset = hook_params.offset

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
                gpf = torch.clamp(
                    self.gamma
                    + self.lambd * (1 - torch.sqrt(torch.var(out_grad) / self.var_a)),
                    min=0,
                    max=1,
                )
            if self.hook_cfg in [
                'vit_base_patch16_224',
                'visformer_small',
                'pit_b_224',
            ]:
                b, c, h, w = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(b, c, h * w)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_h = min_all // h
                min_all_w = min_all % h

                out_grad[:, range(c), max_all_h, :] *= gpf
                out_grad[:, range(c), :, max_all_w] *= gpf
                out_grad[:, range(c), min_all_h, :] *= gpf
                out_grad[:, range(c), :, min_all_w] *= gpf

            if self.hook_cfg in ['cait_s24_224']:
                b, h, w, c = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(b, h * w, c)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
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
                gpf = torch.clamp(
                    self.gamma
                    + self.lambd * (1 - torch.sqrt(torch.var(out_grad) / self.var_a)),
                    min=0,
                    max=1,
                )
            b, h, w, c = grad_in[0].shape
            out_grad_cpu = out_grad.data.clone().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0, :, 0, :], axis=0)
            min_all = np.argmin(out_grad_cpu[0, :, 0, :], axis=0)

            out_grad[:, max_all, :, range(c)] *= gpf
            out_grad[:, min_all, :, range(c)] *= gpf

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
                gpf = torch.clamp(
                    self.gamma
                    + self.lambd * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv)),
                    min=0,
                    max=1,
                )
            out_grad[:] *= gpf
            self.var_qkv = torch.var(out_grad)
            return (out_grad, grad_in[1], grad_in[2])

        def v_att(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            grad_in = (grad_in[0].unsqueeze(0),) + grad_in[1:]

            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            if self.var_qkv.item() == 0:
                gpf = self.gamma
            else:
                gpf = torch.clamp(
                    self.gamma
                    + self.lambd * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv)),
                    min=0,
                    max=1,
                )

            if self.hook_cfg in ['visformer_small']:
                b, c, h, w = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(b, c, h * w)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_h = min_all // h
                min_all_w = min_all % h

                out_grad[:, range(c), max_all_h, max_all_w] *= gpf
                out_grad[:, range(c), min_all_h, min_all_w] *= gpf

            if self.hook_cfg in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

                out_grad[:, max_all, range(c)] *= gpf
                out_grad[:, min_all, range(c)] *= gpf

            out_grad = out_grad.squeeze(0)

            self.var_qkv = torch.var(out_grad)
            # return (out_grad, grad_in[1])
            return (out_grad,) + tuple(grad_in[1:])

        def mlp_att(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            grad_in = (grad_in[0].unsqueeze(0),) + grad_in[1:]

            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[2]
            out_grad = mask * grad_in[0][:]
            if self.var_mlp.item() == 0:
                gpf = self.gamma
            else:
                gpf = torch.clamp(
                    self.gamma
                    + self.lambd * (1 - torch.sqrt(torch.var(out_grad) / self.var_mlp)),
                    min=0,
                    max=1,
                )

            if self.hook_cfg in ['visformer_small']:
                b, c, h, w = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(b, c, h * w)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_h = min_all // h
                min_all_w = min_all % h
                out_grad[:, range(c), max_all_h, max_all_w] *= gpf
                out_grad[:, range(c), min_all_h, min_all_w] *= gpf

            if self.hook_cfg in [
                'vit_base_patch16_224',
                'pit_b_224',
                'cait_s24_224',
                'resnetv2_101',
            ]:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

                out_grad[:, max_all, range(c)] *= gpf
                out_grad[:, min_all, range(c)] *= gpf

            out_grad = out_grad.squeeze(0)

            self.var_mlp = torch.var(out_grad)

            return (out_grad,) + tuple(grad_in[1:])

        def mid_feats_hook(m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
            self.mid_feats = o.clone()

        def mid_grads_hook(m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
            self.mid_grads = o[0].clone()

        # fmt: off
        feature_grad_hook_cfg = {
            'vit_base_patch16_224': 'blocks.10',
            'deit_base_distilled_patch16_224': 'blocks.10',
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
            'deit_base_distilled_patch16_224': [
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
        module.register_backward_hook(mid_grads_hook)

        # Register attention hooks
        for hook_func, layers in attention_hook_cfg[self.hook_cfg]:
            for layer in layers:
                module = rgetattr(self.model, layer)
                module.register_backward_hook(hook_func)


if __name__ == '__main__':
    from torchattack.eval import run_attack

    run_attack(
        ATT,
        model_name='timm/vit_base_patch16_224',
        victim_model_names=['timm/cait_s24_224', 'timm/visformer_small', 'resnet50'],
        # save_adv_batch=6,
        batch_size=2,
        max_samples=10,
    )
