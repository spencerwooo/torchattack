import importlib.util
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# from torchattack._rgetattr import rgetattr
from torchattack.base import Attack


class VDC(Attack):
    """VDC (Virtual Dense Connection) attack for ViTs.

    From the paper: 'Improving the Adversarial Transferability of Vision Transformers
    with Virtual Dense Connection'
    https://ojs.aaai.org/index.php/AAAI/article/view/28541

    Args:
        model: The model to attack.
        model_name: The name of the model.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
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
        model: nn.Module,
        model_name: str,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        sample_num_batches: int = 130,
        lambd: float = 0.1,
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

        self.sample_num_batches = sample_num_batches
        self.lambd = lambd

        # Default (3, 224, 224) image with ViT-B/16 16x16 patches
        self.max_num_batches = int((224 / 16) ** 2)
        self.crop_length = 16

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        self.record_grad = []
        self.record_grad_mlp = []
        ###############
        self.attn_record = []
        self.mlp_record = []
        self.attn_add = []
        self.mlp_add = []
        self.norm_list = []
        self.stage = []
        self.attn_block = 0
        self.mlp_block = 0
        self.hooks = []
        self.skip_record = []
        self.skip_add = []
        self.skip_block = 0

        assert self.sample_num_batches <= self.max_num_batches

        # self._register_model_hooks()

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

        # Perform VDC
        for _ in range(self.steps):
            self.attn_record = []
            self.attn_add = []

            self.mlp_record = []
            self.mlp_add = []

            self.skip_record = []
            self.skip_add = []

            self.mlp_block = 0
            self.attn_block = 0
            self.skip_block = 0
            self._register_model_hooks(add=False)

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

            self.mlp_block = 0
            self.attn_block = 0
            self.skip_block = 0
            self._register_model_hooks(add=True)

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

    def _register_model_hooks(self, add: bool = False):
        def mlp_record_vit_stage(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            # ablation
            grad_record = (
                grad_in[0].data.cpu().numpy() * 0.1 * (0.5 ** (self.mlp_block))
            )
            # grad_record = grad_in[0].data.cpu().numpy()
            if self.mlp_block == 0:
                grad_add = np.zeros_like(grad_record)
                # ablation
                grad_add[:, 0, :] = self.norm_list[:, 0, :] * 0.1 * (0.5)
                # grad_add[:,0,:] = self.norm[:,0,:]
                self.mlp_add.append(grad_add)
                self.mlp_record.append(grad_record + grad_add)
            else:
                self.mlp_add.append(self.mlp_record[-1])
                total_mlp = self.mlp_record[-1] + grad_record
                self.mlp_record.append(total_mlp)
            self.mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])

        def mlp_add_vit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            # mask_0 = torch.zeros_like(grad_in[0])
            out_grad = mask * grad_in[0][:]
            # out_grad = torch.where(grad_in[0][:] > 0, mask * grad_in[0][:], mask_0 * grad_in[0][:])
            out_grad += torch.tensor(self.mlp_add[self.mlp_block]).cuda()
            self.mlp_block += 1
            return (out_grad, grad_in[1], grad_in[2])

        def attn_record_vit_stage(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            grad_record = (
                grad_in[0].data.cpu().numpy() * 0.1 * (0.5 ** (self.attn_block))
            )
            # grad_record = grad_in[0].data.cpu().numpy()
            if self.attn_block == 0:
                self.attn_add.append(np.zeros_like(grad_record))
                self.attn_record.append(grad_record)
            else:
                self.attn_add.append(self.attn_record[-1])
                total_attn = self.attn_record[-1] + grad_record
                self.attn_record.append(total_attn)

            self.attn_block += 1
            return (out_grad,)

        def attn_add_vit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            # mask_0 = torch.zeros_like(grad_in[0])
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            # out_grad = torch.where(grad_in[0][:] > 0, mask * grad_in[0][:], mask_0 * grad_in[0][:])
            out_grad += torch.tensor(self.attn_add[self.attn_block]).cuda()
            self.attn_block += 1
            return (out_grad,)

        def norm_record_vit(module, grad_in, grad_out, gamma):
            grad_record = grad_in[0].data.cpu().numpy()
            mask = torch.ones_like(grad_in[0]) * gamma
            self.norm_list = grad_record
            return grad_in

        # vit
        mlp_record_func_vit = partial(mlp_record_vit_stage, gamma=1.0)
        norm_record_func_vit = partial(norm_record_vit, gamma=1.0)
        mlp_add_func_vit = partial(mlp_add_vit, gamma=0.5)
        attn_record_func_vit = partial(attn_record_vit_stage, gamma=1.0)
        attn_add_func_vit = partial(attn_add_vit, gamma=0.25)

        if not add:
            if self.model_name in [
                'vit_base_patch16_224',
                'deit_base_distilled_patch16_224',
            ]:
                hook = self.model.norm.register_backward_hook(norm_record_func_vit)
                self.hooks.append(hook)
                for i in range(12):
                    hook = self.model.blocks[i].norm2.register_backward_hook(
                        mlp_record_func_vit
                    )
                    self.hooks.append(hook)
                    hook = self.model.blocks[i].attn.attn_drop.register_backward_hook(
                        attn_record_func_vit
                    )
                    self.hooks.append(hook)
        else:
            if self.model_name in [
                'vit_base_patch16_224',
                'deit_base_distilled_patch16_224',
            ]:
                for i in range(12):
                    hook = self.model.blocks[i].norm2.register_backward_hook(
                        mlp_add_func_vit
                    )
                    self.hooks.append(hook)
                    hook = self.model.blocks[i].attn.attn_drop.register_backward_hook(
                        attn_add_func_vit
                    )
                    self.hooks.append(hook)


if __name__ == '__main__':
    from torchattack.eval import run_attack

    run_attack(
        VDC,
        attack_cfg={'model_name': 'vit_base_patch16_224'},
        model_name='vit_base_patch16_224',
        victim_model_names=['cait_s24_224', 'visformer_small'],
        batch_size=4,
        from_timm=False,
    )
