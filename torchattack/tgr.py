from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack_model import AttackModel
from torchattack.base import Attack


class TGR(Attack):
    """TGR attack for ViTs (Token Gradient Regularization).

    From the paper: 'Transferable Adversarial Attacks on Vision Transformers with Token
    Gradient Regularization'
    https://arxiv.org/abs/2303.15754

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        model_name: The name of the model. Supported models are:
            * 'vit_base_patch16_224'
            * 'deit_base_distilled_patch16_224'
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
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None = None,
        model_name: str = '',
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        # Surrogate ViT for VDC must be `timm` models or models that have the same
        # structure and same implementation/definition as `timm` models.
        super().__init__(model, normalize, device)

        if model_name:
            # Explicit model_name takes precedence over model.model_name
            self.model_name = model_name
        elif hasattr(model, 'model_name'):
            # If model is initialized via `torchattack.eval.AttackModel`, the model_name
            # is automatically attached to the model during instantiation.
            self.model_name = model.model_name
        else:
            raise ValueError('`model_name` must be explicitly provided.')

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        # Register hooks
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
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

            for hook in self.hooks:
                hook.remove()

        return x + delta

    def _register_tgr_model_hooks(self):
        def attn_tgr(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name in [
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
                out_grad[:, range(c), max_all_h, :] = 0.0
                out_grad[:, range(c), :, max_all_w] = 0.0
                out_grad[:, range(c), min_all_h, :] = 0.0
                out_grad[:, range(c), :, min_all_w] = 0.0

            if self.model_name == 'cait_s24_224':
                b, h, w, c = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(b, h * w, c)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
                min_all_h = min_all // h
                min_all_w = min_all % h

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
            out_grad_cpu = out_grad.data.clone().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0, :, 0, :], axis=0)
            min_all = np.argmin(out_grad_cpu[0, :, 0, :], axis=0)

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
            # show diff between high and low PyTorch version
            # print('v len(grad_in)',len(grad_in))
            # high, 1
            # low, 2

            # print('v grad_in[0].shape',grad_in[0].shape)
            # high, torch.Size([197, 2304])
            # low, torch.Size([1, 197, 2304])
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)  # type: ignore
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)  # type: ignore

            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if self.model_name == 'visformer_small':
                b, c, h, w = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(b, c, h * w)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_h = min_all // h
                min_all_w = min_all % h
                out_grad[:, range(c), max_all_h, max_all_w] = 0.0
                out_grad[:, range(c), min_all_h, min_all_w] = 0.0

            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

                out_grad[:, max_all, range(c)] = 0.0
                out_grad[:, min_all, range(c)] = 0.0

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)

            # return (out_grad, grad_in[1])
            return (out_grad,) + tuple(grad_in[1:])

        def mlp_tgr(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)  # type: ignore
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)  # type: ignore

            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name == 'visformer_small':
                b, c, h, w = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(b, c, h * w)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_h = max_all // h
                max_all_w = max_all % h
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_h = min_all // h
                min_all_w = min_all % h
                out_grad[:, range(c), max_all_h, max_all_w] = 0.0
                out_grad[:, range(c), min_all_h, min_all_w] = 0.0
            if self.model_name in [
                'vit_base_patch16_224',
                'pit_b_224',
                'cait_s24_224',
                'resnetv2_101',
            ]:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
                out_grad[:, max_all, range(c)] = 0.0
                out_grad[:, min_all, range(c)] = 0.0

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)

            return (out_grad,) + tuple(grad_in[1:])

        attn_tgr_hook = partial(attn_tgr, gamma=0.25)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=0.25)
        v_tgr_hook = partial(v_tgr, gamma=0.75)
        q_tgr_hook = partial(q_tgr, gamma=0.75)
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

        # fmt: off
        supported_vit_cfg = {
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

        assert self.model_name in supported_vit_cfg

        for hook_func, layers in supported_vit_cfg[self.model_name]:
            for layer in layers:
                module = rgetattr(self.model, layer)
                hook = module.register_backward_hook(hook_func)
                self.hooks.append(hook)


if __name__ == '__main__':
    from torchattack.eval import run_attack

    run_attack(
        TGR,
        model_name='vit_base_patch16_224',
        victim_model_names=['cait_s24_224', 'visformer_small'],
        max_samples=24,
        batch_size=4,
        from_timm=True,
    )