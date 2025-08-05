from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class SGM(Attack):
    """The SGM (Skip Gradient Method) attack.

    > From the paper: [Skip Connections Matter: On the Transferability of Adversarial
    Examples Generated with ResNets](https://arxiv.org/abs/2002.05990).

    Note:
        `hook_cfg` should match the passed `model` name. If `model` is not initialized
        via `torchattack.AttackModel`, then `hook_cfg` must be specified explicitly.
        Supported models include: `resnet18`, `resnet34`, `resnet50`, `resnet101`,
        `resnet152`, `densenet121`, `densenet161`, `densenet169`, `densenet201`.

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        hook_cfg: Config used for applying hooks to the model. Supported values:
            `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `densenet121`,
            `densenet161`, `densenet169`, `densenet201`.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        gamma: Decay factor for the gradient from residual modules. Defaults to 0.2.
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
        gamma: float = 0.2,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        if hook_cfg:
            # Explicit config name takes precedence over inferred model.model_name
            self.hook_cfg = hook_cfg
        elif isinstance(model, AttackModel):
            # If model is initialized via `torchattack.AttackModel`, the model_name
            # is automatically attached to the model during instantiation.
            self.hook_cfg = model.model_name

        # Delay initialization to avoid overriding the model's `model_name` attribute
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register hooks to the model for SGM."""

        is_resnet = 'resnet' in self.hook_cfg
        is_densenet = 'densenet' in self.hook_cfg

        if not (is_resnet or is_densenet):
            raise ValueError(f'Unsupported model hook config: {self.hook_cfg}')

        # Adjust gamma for models with multiple ReLUs in their skip/conv blocks.
        # ResNet-50/101/152 (2 ReLUs in bottleneck) and DenseNets (2 ReLUs in dense layer)
        # typically require gamma^0.5, while ResNet-18/34 (1 ReLU in basic block path) use gamma.
        deep_resnets = ['resnet50', 'resnet101', 'resnet152']
        if (is_resnet and self.hook_cfg in deep_resnets) or is_densenet:
            self.gamma = np.power(self.gamma, 0.5)

        backward_hook_sgm = self._backward_hook(self.gamma)

        for name, mod in self.model.named_modules():
            if is_resnet:
                # Apply SGM hook to specific ReLUs in ResNet.
                # The condition `'0.relu' not in name` is from the original implementation.
                if 'relu' in name and '0.relu' not in name:
                    mod.register_backward_hook(backward_hook_sgm)

                # Apply gradient normalization hook to ResNet layer modules.
                # Targets modules like 'layerX.Y' (e.g., 'layer1.0', 'layer2.1').
                parts = name.split('.')
                if len(parts) >= 2 and 'layer' in parts[-2]:
                    mod.register_backward_hook(self._backward_hook_normalized)

            elif is_densenet:
                # Apply SGM hook to ReLUs in DenseNet, excluding those in transition layers.
                if 'relu' in name and 'transition' not in name:
                    mod.register_backward_hook(backward_hook_sgm)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform SGM on a batch of images.

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

        # Perform SGM
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

    def _backward_hook(self, gamma: float) -> Callable:
        def hook(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...],
            grad_out: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            if isinstance(module, nn.ReLU):
                return (gamma * grad_in[0],)
            return grad_in

        return hook

    def _backward_hook_normalized(
        self,
        module: nn.Module,
        grad_in: tuple[torch.Tensor, ...],
        grad_out: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        std = torch.std(grad_in[0])
        return (grad_in[0] / std,)


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=SGM,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
