from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack(category='GRADIENT_VIT')
class TGR(Attack):
    """TGR attack for ViTs (Token Gradient Regularization).

    > From the paper: [Transferable Adversarial Attacks on Vision Transformers with
    Token Gradient Regularization](https://arxiv.org/abs/2303.15754).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        hook_cfg: Config used for applying hooks to the model. Supported values:
            `vit_base_patch16_224`, `deit_base_distilled_patch16_224`, `pit_b_224`,
            `cait_s24_224`, `visformer_small`.
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
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        # Register hooks
        self._register_tgr_model_hooks()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform TGR on a batch of images.

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

        # Perform TGR
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

    def _register_tgr_model_hooks(self) -> None:
        def attn_tgr(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.hook_cfg in [
                'vit_base_patch16_224',
                'visformer_small',
                'pit_b_224',
            ]:
                b, c, h, w = grad_in[0].shape
                out_grad_reshaped = out_grad.view(b, c, h * w)
                max_all = torch.argmax(out_grad_reshaped[0], dim=1)
                max_all_h = max_all // w
                max_all_w = max_all % w
                min_all = torch.argmin(out_grad_reshaped[0], dim=1)
                min_all_h = min_all // w
                min_all_w = min_all % w
                out_grad[:, range(c), max_all_h, :] = 0.0
                out_grad[:, range(c), :, max_all_w] = 0.0
                out_grad[:, range(c), min_all_h, :] = 0.0
                out_grad[:, range(c), :, min_all_w] = 0.0

            if self.hook_cfg == 'cait_s24_224':
                b, h, w, c = grad_in[0].shape
                out_grad_reshaped = out_grad.view(b, h * w, c)
                max_all = torch.argmax(out_grad_reshaped[0], dim=0)
                max_all_h = max_all // w
                max_all_w = max_all % w
                min_all = torch.argmin(out_grad_reshaped[0], dim=0)
                min_all_h = min_all // w
                min_all_w = min_all % w

                out_grad[:, max_all_h, :, range(c)] = 0.0
                out_grad[:, :, max_all_w, range(c)] = 0.0
                out_grad[:, min_all_h, :, range(c)] = 0.0
                out_grad[:, :, min_all_w, range(c)] = 0.0

            return (out_grad,)

        def attn_cait_tgr(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            b, h, w, c = grad_in[0].shape
            out_grad_reshaped = out_grad.view(b, h * w, c)
            max_all = torch.argmax(out_grad_reshaped[0, :, :], dim=0)
            min_all = torch.argmin(out_grad_reshaped[0, :, :], dim=0)

            out_grad[:, max_all, :, range(c)] = 0.0
            out_grad[:, min_all, :, range(c)] = 0.0
            return (out_grad,)

        def q_tgr(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad[:] = 0.0
            return (out_grad, grad_in[1], grad_in[2])

        def v_tgr(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            is_dim_extra = False
            if len(grad_in[0].shape) == 2:
                is_dim_extra = True
                grad_in = (grad_in[0].unsqueeze(0),) + grad_in[1:]

            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if self.hook_cfg == 'visformer_small':
                b, c, h, w = grad_in[0].shape
                out_grad_reshaped = out_grad.view(b, c, -1)
                max_all = torch.argmax(out_grad_reshaped[0], dim=1)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = torch.argmin(out_grad_reshaped[0], dim=1)
                min_all_h = min_all // h
                min_all_w = min_all % h
                out_grad[:, range(c), max_all_h, max_all_w] = 0.0
                out_grad[:, range(c), min_all_h, min_all_w] = 0.0

            if self.hook_cfg in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                max_all = torch.argmax(out_grad[0], dim=0)
                min_all = torch.argmin(out_grad[0], dim=0)
                out_grad[:, max_all, range(c)] = 0.0
                out_grad[:, min_all, range(c)] = 0.0

            if is_dim_extra:
                out_grad = out_grad.squeeze(0)

            # return (out_grad, grad_in[1])
            return (out_grad,) + tuple(grad_in[1:])

        def mlp_tgr(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            is_dim_extra = False
            if len(grad_in[0].shape) == 2:
                is_dim_extra = True
                grad_in = (grad_in[0].unsqueeze(0),) + grad_in[1:]

            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.hook_cfg == 'visformer_small':
                b, c, h, w = grad_in[0].shape
                out_grad_reshaped = out_grad.view(b, c, -1)
                max_all = torch.argmax(out_grad_reshaped[0], dim=1)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = torch.argmin(out_grad_reshaped[0], dim=1)
                min_all_h = min_all // h
                min_all_w = min_all % h
                out_grad[:, range(c), max_all_h, max_all_w] = 0.0
                out_grad[:, range(c), min_all_h, min_all_w] = 0.0
            if self.hook_cfg in [
                'vit_base_patch16_224',
                'pit_b_224',
                'cait_s24_224',
                'resnetv2_101',
            ]:
                c = grad_in[0].shape[2]
                max_all = torch.argmax(out_grad[0], dim=0)
                min_all = torch.argmin(out_grad[0], dim=0)
                out_grad[:, max_all, range(c)] = 0.0
                out_grad[:, min_all, range(c)] = 0.0

            if is_dim_extra:
                out_grad = out_grad.squeeze(0)

            return (out_grad,) + tuple(grad_in[1:])

        attn_tgr_hook = partial(attn_tgr, gamma=0.25)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=0.25)
        v_tgr_hook = partial(v_tgr, gamma=0.75)
        q_tgr_hook = partial(q_tgr, gamma=0.75)
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

        # fmt: off
        supported_hook_cfg = {
            'vit_base_patch16_224': [
                (attn_tgr_hook, [f'blocks.{i}.attn.attn_drop' for i in range(12)]),
                (v_tgr_hook, [f'blocks.{i}.attn.qkv' for i in range(12)]),
                (mlp_tgr_hook, [f'blocks.{i}.mlp' for i in range(12)]),
            ],
            'deit_base_distilled_patch16_224': [
                (attn_tgr_hook, [f'blocks.{i}.attn.attn_drop' for i in range(12)]),
                (v_tgr_hook, [f'blocks.{i}.attn.qkv' for i in range(12)]),
                (mlp_tgr_hook, [f'blocks.{i}.mlp' for i in range(12)]),
            ],
            'pit_b_224': [
                (attn_tgr_hook, [f'transformers.{tid}.blocks.{i}.attn.attn_drop' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
                (v_tgr_hook, [f'transformers.{tid}.blocks.{i}.attn.qkv' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
                (mlp_tgr_hook, [f'transformers.{tid}.blocks.{i}.mlp' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
            ],
            'cait_s24_224': [
                (attn_tgr_hook, [f'blocks.{i}.attn.attn_drop' for i in range(24)]),
                (v_tgr_hook, [f'blocks.{i}.attn.qkv' for i in range(24)]),
                (mlp_tgr_hook, [f'blocks.{i}.mlp' for i in range(24)]),
                (attn_cait_tgr_hook, [f'blocks_token_only.{i}.attn.attn_drop' for i in range(0, 2)]),
                (q_tgr_hook, [f'blocks_token_only.{i}.attn.q' for i in range(0, 2)]),
                (v_tgr_hook, [f'blocks_token_only.{i}.attn.k' for i in range(0, 2)]),
                (v_tgr_hook, [f'blocks_token_only.{i}.attn.v' for i in range(0, 2)]),
                (mlp_tgr_hook, [f'blocks_token_only.{i}.mlp' for i in range(0, 2)]),
            ],
            'visformer_small': [
                (attn_tgr_hook, [f'stage2.{i}.attn.attn_drop' for i in range(4)]),
                (v_tgr_hook, [f'stage2.{i}.attn.qkv' for i in range(4)]),
                (mlp_tgr_hook, [f'stage2.{i}.mlp' for i in range(4)]),
                (attn_tgr_hook, [f'stage3.{i}.attn.attn_drop' for i in range(4)]),
                (v_tgr_hook, [f'stage3.{i}.attn.qkv' for i in range(4)]),
                (mlp_tgr_hook, [f'stage3.{i}.mlp' for i in range(4)]),
            ],
        }
        # fmt: on

        if self.hook_cfg not in supported_hook_cfg:
            from warnings import warn

            warn(
                f'Hook config specified (`{self.hook_cfg}`) is not supported. '
                'Falling back to default (`vit_base_patch16_224`). '
                'This MAY NOT be intended.',
                stacklevel=2,
            )
            self.hook_cfg = 'vit_base_patch16_224'

        for hook_func, layers in supported_hook_cfg[self.hook_cfg]:
            for layer in layers:
                module = rgetattr(self.model, layer)
                module.register_backward_hook(hook_func)


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    model_names = [
        'timm/vit_base_patch16_224',
        'timm/pit_b_224',
        'timm/cait_s24_224',
        'timm/visformer_small',
    ]
    for name in model_names:
        run_attack(
            TGR,
            model_name=name,
            victim_model_names=[
                'timm/vit_base_patch16_224',
                'timm/pit_b_224',
                'timm/cait_s24_224',
                'timm/visformer_small',
            ],
            batch_size=4,
        )
