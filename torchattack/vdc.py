from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torchattack._attack import Attack
from torchattack._rgetattr import rgetattr
from torchattack.attack_model import AttackModel


class VDC(Attack):
    """VDC (Virtual Dense Connection) attack for ViTs.

    From the paper: 'Improving the Adversarial Transferability of Vision Transformers
    with Virtual Dense Connection'
    https://ojs.aaai.org/index.php/AAAI/article/view/28541

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        hook_cfg: Config used for applying hooks to the model. Supported configs are:
            * 'vit_base_patch16_224'
            * 'deit_base_distilled_patch16_224'
            * 'pit_b_224'
            * 'visformer_small'
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Momentum decay factor. Defaults to 1.0.
        sample_num_batches: Number of batches to sample. Defaults to 130.
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
        # lambd: float = 0.1,
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

        self.sample_num_batches = sample_num_batches
        # self.lambd = lambd

        # Default (3, 224, 224) image with ViT-B/16 16x16 patches
        self.max_num_batches = int((224 / 16) ** 2)
        self.crop_length = 16

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        # Global hooks and attack stage state for VDC
        self.stage: list[np.ndarray] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

        assert self.sample_num_batches <= self.max_num_batches

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform VDC on a batch of images.

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

        class GradientRecorder:
            """Gradient recorder for attention and MLP blocks."""

            def __init__(self):
                self.grad_records = []
                self.grad_additions = []

        # Perform VDC
        for _ in range(self.steps):
            # Initialize gradient recorders
            self.attn_recorder = GradientRecorder()
            self.mlp_recorder = GradientRecorder()

            # Stage 1: Record gradients
            self.current_mlp_block = 0
            self.current_attn_block = 0
            self._register_model_hooks(add_grad_mode=False)

            # Compute loss
            outs = self.model(self.normalize(x + delta))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

            for hook in self.hooks:
                hook.remove()

            # Stage 2: Update gradients by adding recorded gradients
            self.current_mlp_block = 0
            self.current_attn_block = 0
            self._register_model_hooks(add_grad_mode=True)

            # Compute loss 2nd time
            outs = self.model(self.normalize(x + delta))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient 2nd time
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

    def _register_model_hooks(self, add_grad_mode: bool = False) -> None:
        """Register hooks to either record or add gradients during the backward pass.

        Args:
            grad_add_hook: If False, register hooks to record gradients. If True,
                register hooks to modify the gradients by adding pre-recorded gradients
                during the backward pass.
        """

        def mlp_record_vit_stage(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            # ablation
            grad_record = (
                grad_in[0].data.cpu().numpy() * 0.1 * (0.5 ** (self.current_mlp_block))
            )
            # grad_record = grad_in[0].data.cpu().numpy()
            if self.current_mlp_block == 0:
                grad_add = np.zeros_like(grad_record)
                # ablation
                grad_add[:, 0, :] = self.norm_list[:, 0, :] * 0.1 * (0.5)
                # grad_add[:,0,:] = self.norm[:,0,:]
                self.mlp_recorder.grad_additions.append(grad_add)
                self.mlp_recorder.grad_records.append(grad_record + grad_add)
            else:
                self.mlp_recorder.grad_additions.append(
                    self.mlp_recorder.grad_records[-1]
                )
                total_mlp = self.mlp_recorder.grad_records[-1] + grad_record
                self.mlp_recorder.grad_records.append(total_mlp)
            self.current_mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])

        def mlp_add_vit(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            # grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            # mask_0 = torch.zeros_like(grad_in[0])
            out_grad = mask * grad_in[0][:]
            # out_grad = torch.where(
            #     grad_in[0][:] > 0, mask * grad_in[0][:], mask_0 * grad_in[0][:]
            # )
            out_grad += torch.tensor(
                self.mlp_recorder.grad_additions[self.current_mlp_block],
                device=grad_in[0].device,
            )
            self.current_mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])

        def attn_record_vit_stage(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            grad_record = (
                grad_in[0].data.cpu().numpy() * 0.1 * (0.5 ** (self.current_attn_block))
            )
            # grad_record = grad_in[0].data.cpu().numpy()
            if self.current_attn_block == 0:
                self.attn_recorder.grad_additions.append(np.zeros_like(grad_record))
                self.attn_recorder.grad_records.append(grad_record)
            else:
                self.attn_recorder.grad_additions.append(
                    self.attn_recorder.grad_records[-1]
                )
                total_attn = self.attn_recorder.grad_records[-1] + grad_record
                self.attn_recorder.grad_records.append(total_attn)

            self.current_attn_block += 1
            return (out_grad,)

        def attn_add_vit(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            # grad_record = grad_in[0].data.cpu().numpy()
            # mask_0 = torch.zeros_like(grad_in[0])
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            # out_grad = torch.where(
            #     grad_in[0][:] > 0, mask * grad_in[0][:], mask_0 * grad_in[0][:]
            # )
            out_grad += torch.tensor(
                self.attn_recorder.grad_additions[self.current_attn_block],
                device=grad_in[0].device,
            )
            self.current_attn_block += 1
            return (out_grad,)

        def norm_record_vit(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            grad_record = grad_in[0].data.cpu().numpy()
            # mask = torch.ones_like(grad_in[0]) * gamma
            self.norm_list = grad_record
            return grad_in

        # pit
        def pool_record_pit(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            grad_add = grad_in[0].data
            b, c, h, w = grad_add.shape
            grad_add = grad_add.reshape((b, c, h * w)).transpose(1, 2)
            self.stage.append(grad_add.cpu().numpy())
            return grad_in

        def mlp_record_pit_stage(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.current_mlp_block < 4:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.03
                    * (0.5 ** (self.current_mlp_block))
                )
                if self.current_mlp_block == 0:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:, 0, :] = self.norm_list[:, 0, :] * 0.03 * (0.5)
                    self.mlp_recorder.grad_additions.append(grad_add)
                    self.mlp_recorder.grad_records.append(grad_record + grad_add)
                else:
                    self.mlp_recorder.grad_additions.append(
                        self.mlp_recorder.grad_records[-1]
                    )
                    total_mlp = self.mlp_recorder.grad_records[-1] + grad_record
                    self.mlp_recorder.grad_records.append(total_mlp)
            elif self.current_mlp_block < 10:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.03
                    * (0.5 ** (self.current_mlp_block))
                )
                if self.current_mlp_block == 4:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:, 1:, :] = self.stage[0] * 0.03 * (0.5)
                    self.mlp_recorder.grad_additions.append(grad_add)
                    self.mlp_recorder.grad_records.append(grad_record + grad_add)
                else:
                    self.mlp_recorder.grad_additions.append(
                        self.mlp_recorder.grad_records[-1]
                    )
                    # total_mlp = self.mlp_rec.record[-1] + grad_record
                    total_mlp = self.mlp_recorder.grad_records[-1]
                    self.mlp_recorder.grad_records.append(total_mlp)
            else:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.03
                    * (0.5 ** (self.current_mlp_block))
                )
                if self.current_mlp_block == 10:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:, 1:, :] = self.stage[1] * 0.03 * (0.5)
                    self.mlp_recorder.grad_additions.append(grad_add)
                    self.mlp_recorder.grad_records.append(grad_record + grad_add)
                else:
                    self.mlp_recorder.grad_additions.append(
                        self.mlp_recorder.grad_records[-1]
                    )
                    # total_mlp = self.mlp_rec.record[-1] + grad_record
                    total_mlp = self.mlp_recorder.grad_records[-1]
                    self.mlp_recorder.grad_records.append(total_mlp)
            self.current_mlp_block += 1

            return (out_grad, grad_in[1], grad_in[2])

        def mlp_add_pit(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            # grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(
                self.mlp_recorder.grad_additions[self.current_mlp_block],
                device=grad_in[0].device,
            )
            self.current_mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])

        def attn_record_pit_stage(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.current_attn_block < 4:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.03
                    * (0.5 ** (self.current_attn_block))
                )
                if self.current_attn_block == 0:
                    self.attn_recorder.grad_additions.append(np.zeros_like(grad_record))
                    self.attn_recorder.grad_records.append(grad_record)
                else:
                    self.attn_recorder.grad_additions.append(
                        self.attn_recorder.grad_records[-1]
                    )
                    total_attn = self.attn_recorder.grad_records[-1] + grad_record
                    self.attn_recorder.grad_records.append(total_attn)
            elif self.current_attn_block < 10:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.03
                    * (0.5 ** (self.current_attn_block))
                )
                if self.current_attn_block == 4:
                    self.attn_recorder.grad_additions.append(np.zeros_like(grad_record))
                    self.attn_recorder.grad_records.append(grad_record)
                else:
                    self.attn_recorder.grad_additions.append(
                        self.attn_recorder.grad_records[-1]
                    )
                    # total_attn = self.attn_rec.record[-1] + grad_record
                    total_attn = self.attn_recorder.grad_records[-1]
                    self.attn_recorder.grad_records.append(total_attn)
            else:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.03
                    * (0.5 ** (self.current_attn_block))
                )
                if self.current_attn_block == 10:
                    self.attn_recorder.grad_additions.append(np.zeros_like(grad_record))
                    self.attn_recorder.grad_records.append(grad_record)
                else:
                    self.attn_recorder.grad_additions.append(
                        self.attn_recorder.grad_records[-1]
                    )
                    # total_attn = self.attn_rec.record[-1] + grad_record
                    total_attn = self.attn_recorder.grad_records[-1]
                    self.attn_recorder.grad_records.append(total_attn)
            self.current_attn_block += 1
            return (out_grad,)

        def attn_add_pit(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            # grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(
                self.attn_recorder.grad_additions[self.current_attn_block],
                device=grad_in[0].device,
            )
            self.current_attn_block += 1
            return (out_grad,)

        def norm_record_pit(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            grad_record = grad_in[0].data.cpu().numpy()
            # mask = torch.ones_like(grad_in[0]) * gamma
            self.norm_list = grad_record
            return grad_in

        # visformer
        def pool_record_vis(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            grad_add = grad_in[0].data
            # B,C,H,W = grad_add.shape
            # grad_add = grad_add.reshape((B,C,H*W)).transpose(1,2)
            self.stage.append(grad_add.cpu().numpy())
            return grad_in

        def mlp_record_vis_stage(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.current_mlp_block < 4:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.1
                    * (0.5 ** (self.current_mlp_block))
                )
                if self.current_mlp_block == 0:
                    grad_add = np.zeros_like(grad_record)
                    grad_add[:, 0, :] = self.norm_list[:, 0, :] * 0.1 * (0.5)
                    self.mlp_recorder.grad_additions.append(grad_add)
                    self.mlp_recorder.grad_records.append(grad_record + grad_add)
                else:
                    self.mlp_recorder.grad_additions.append(
                        self.mlp_recorder.grad_records[-1]
                    )
                    total_mlp = self.mlp_recorder.grad_records[-1] + grad_record
                    self.mlp_recorder.grad_records.append(total_mlp)
            else:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.1
                    * (0.5 ** (self.current_mlp_block))
                )
                if self.current_mlp_block == 4:
                    grad_add = np.zeros_like(grad_record)
                    # grad_add[:,1:,:] = self.stage[0]* 0.1*(0.5)
                    self.mlp_recorder.grad_additions.append(grad_add)
                    self.mlp_recorder.grad_records.append(grad_record + grad_add)
                else:
                    self.mlp_recorder.grad_additions.append(
                        self.mlp_recorder.grad_records[-1]
                    )
                    total_mlp = self.mlp_recorder.grad_records[-1] + grad_record
                    self.mlp_recorder.grad_records.append(total_mlp)

            self.current_mlp_block += 1

            return (out_grad, grad_in[1], grad_in[2])

        def mlp_add_vis(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            # grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(
                self.mlp_recorder.grad_additions[self.current_mlp_block],
                device=grad_in[0].device,
            )
            self.current_mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])

        def norm_record_vis(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            grad_record = grad_in[0].data.cpu().numpy()
            # mask = torch.ones_like(grad_in[0]) * gamma
            self.norm_list = grad_record
            return grad_in

        def attn_record_vis_stage(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.current_attn_block < 4:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.1
                    * (0.5 ** (self.current_attn_block))
                )
                if self.current_attn_block == 0:
                    self.attn_recorder.grad_additions.append(np.zeros_like(grad_record))
                    self.attn_recorder.grad_records.append(grad_record)
                else:
                    self.attn_recorder.grad_additions.append(
                        self.attn_recorder.grad_records[-1]
                    )
                    total_attn = self.attn_recorder.grad_records[-1] + grad_record
                    self.attn_recorder.grad_records.append(total_attn)
            else:
                grad_record = (
                    grad_in[0].data.cpu().numpy()
                    * 0.1
                    * (0.5 ** (self.current_attn_block))
                )
                if self.current_attn_block == 4:
                    self.attn_recorder.grad_additions.append(np.zeros_like(grad_record))
                    self.attn_recorder.grad_records.append(grad_record)
                else:
                    self.attn_recorder.grad_additions.append(
                        self.attn_recorder.grad_records[-1]
                    )
                    total_attn = self.attn_recorder.grad_records[-1] + grad_record
                    self.attn_recorder.grad_records.append(total_attn)

            self.current_attn_block += 1
            return (out_grad,)

        def attn_add_vis(
            module: torch.nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
            gamma: float,
        ) -> tuple[torch.Tensor, ...]:
            # grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad += torch.tensor(
                self.attn_recorder.grad_additions[self.current_attn_block],
                device=grad_in[0].device,
            )
            self.current_attn_block += 1
            return (out_grad,)

        # vit
        mlp_record_func_vit = partial(mlp_record_vit_stage, gamma=1.0)
        norm_record_func_vit = partial(norm_record_vit, gamma=1.0)
        mlp_add_func_vit = partial(mlp_add_vit, gamma=0.5)
        attn_record_func_vit = partial(attn_record_vit_stage, gamma=1.0)
        attn_add_func_vit = partial(attn_add_vit, gamma=0.25)

        # pit
        attn_record_func_pit = partial(attn_record_pit_stage, gamma=1.0)
        mlp_record_func_pit = partial(mlp_record_pit_stage, gamma=1.0)
        norm_record_func_pit = partial(norm_record_pit, gamma=1.0)
        pool_record_func_pit = partial(pool_record_pit, gamma=1.0)
        attn_add_func_pit = partial(attn_add_pit, gamma=0.25)
        mlp_add_func_pit = partial(mlp_add_pit, gamma=0.5)
        # mlp_add_func_pit = partial(mlp_add_pit, gamma=0.75)

        # visformer
        attn_record_func_vis = partial(attn_record_vis_stage, gamma=1.0)
        mlp_record_func_vis = partial(mlp_record_vis_stage, gamma=1.0)
        norm_record_func_vis = partial(norm_record_vis, gamma=1.0)
        pool_record_func_vis = partial(pool_record_vis, gamma=1.0)
        attn_add_func_vis = partial(attn_add_vis, gamma=0.25)
        mlp_add_func_vis = partial(mlp_add_vis, gamma=0.5)

        # fmt: off
        # Register hooks for supported models.
        #   * Gradient RECORD mode hooks:
        grad_record_hook_cfg = {
            'vit_base_patch16_224': [
                (norm_record_func_vit, ['norm']),
                (mlp_record_func_vit, [f'blocks.{i}.norm2' for i in range(12)]),
                (attn_record_func_vit, [f'blocks.{i}.attn.attn_drop' for i in range(12)]),
            ],
            'deit_base_distilled_patch16_224': [
                (norm_record_func_vit, ['norm']),
                (mlp_record_func_vit, [f'blocks.{i}.norm2' for i in range(12)]),
                (attn_record_func_vit, [f'blocks.{i}.attn.attn_drop' for i in range(12)]),
            ],
            'pit_b_224': [
                (norm_record_func_pit, ['norm']),
                (attn_record_func_pit, [f'transformers.{tid}.blocks.{i}.attn.attn_drop' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
                (mlp_record_func_pit, [f'transformers.{tid}.blocks.{i}.norm2' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
                (pool_record_func_pit, ['transformers.1.pool', 'transformers.2.pool']),
            ],
            'visformer_small': [
                (norm_record_func_vis, ['norm']),
                (attn_record_func_vis, [f'stage2.{i}.attn.attn_drop' for i in range(4)] + [f'stage3.{i}.attn.attn_drop' for i in range(4)]),
                (mlp_record_func_vis, [f'stage2.{i}.norm2' for i in range(4)] + [f'stage3.{i}.norm2' for i in range(4)]),
                (pool_record_func_vis, ['patch_embed2', 'patch_embed3']),
            ],
        }
        #   * Gradient ADD mode hooks:
        grad_add_hook_cfg = {
            'vit_base_patch16_224': [
                (mlp_add_func_vit, [f'blocks.{i}.norm2' for i in range(12)]),
                (attn_add_func_vit, [f'blocks.{i}.attn.attn_drop' for i in range(12)]),
            ],
            'deit_base_distilled_patch16_224': [
                (mlp_add_func_vit, [f'blocks.{i}.norm2' for i in range(12)]),
                (attn_add_func_vit, [f'blocks.{i}.attn.attn_drop' for i in range(12)]),
            ],
            'pit_b_224': [
                (attn_add_func_pit, [f'transformers.{tid}.blocks.{i}.attn.attn_drop' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
                (mlp_add_func_pit, [f'transformers.{tid}.blocks.{i}.norm2' for tid, bid in enumerate([3, 6, 4]) for i in range(bid)]),
            ],
            'visformer_small': [
                (attn_add_func_vis, [f'stage2.{i}.attn.attn_drop' for i in range(4)] + [f'stage3.{i}.attn.attn_drop' for i in range(4)]),
                (mlp_add_func_vis, [f'stage2.{i}.norm2' for i in range(4)] + [f'stage3.{i}.norm2' for i in range(4)]),

            ],
        }
        # fmt: on

        active_hook_cfg = grad_add_hook_cfg if add_grad_mode else grad_record_hook_cfg

        if self.hook_cfg not in active_hook_cfg:
            from warnings import warn

            warn(
                f'Hook config specified (`{self.hook_cfg}`) is not supported. '
                'Falling back to default (`vit_base_patch16_224`). '
                'This MAY NOT be intended.',
                stacklevel=2,
            )
            self.hook_cfg = 'vit_base_patch16_224'

        for hook_func, layers in active_hook_cfg[self.hook_cfg]:
            for layer in layers:
                module = rgetattr(self.model, layer)
                hook = module.register_backward_hook(hook_func)
                self.hooks.append(hook)


if __name__ == '__main__':
    from torchattack.eval import run_attack

    run_attack(
        VDC,
        model_name='timm/pit_b_224',
        victim_model_names=['timm/cait_s24_224', 'timm/visformer_small'],
    )
