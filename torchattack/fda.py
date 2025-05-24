from typing import Any, Callable

import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class FDA(Attack):
    """The FDA (Feature Disruptive Attack).

    > From the paper: [FDA: Feature Disruptive Attack](https://arxiv.org/abs/1909.04385).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        random_start: Start from random uniform perturbation. Defaults to True.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
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
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.random_start = random_start
        self.alpha = alpha
        self.decay = decay
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Store intermediate features
        self.features: dict[str, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register hooks to capture intermediate features during forward pass."""

        def hook_fn(
            activation_layer_path: str,
        ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
            def hook(
                m: nn.Module, i: tuple[torch.Tensor, ...], o: torch.Tensor
            ) -> None:
                self.features[activation_layer_path] = o

            return hook

        # Register hooks for ReLU layers and other important layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                # Skip certain layers based on naming patterns
                if any(skip in name.lower() for skip in ['conv', 'mixed']):
                    continue
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                # Register hooks for pooling layers
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform FDA on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        g = torch.zeros_like(x)
        batch_size = x.size(0)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Initialize perturbation
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            delta = torch.clamp(delta, -self.eps, self.eps)
            delta.requires_grad_(True)
        else:
            delta = torch.zeros_like(x, requires_grad=True)

        # Perform FDA
        for _ in range(self.steps):
            # Clear previous features
            self.features.clear()

            # Create adversarial examples
            xadv = x + delta

            # Concatenate original and adversarial examples
            concat_x = torch.cat([x, xadv], dim=0)

            # Forward pass to collect intermediate features
            _ = self.model(self.normalize(concat_x))

            # Compute FDA loss
            loss = self._get_fda_loss(batch_size)

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

            # Zero gradients
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta

    def _get_fda_loss(self, batch_size: int) -> torch.Tensor:
        """Compute FDA loss based on intermediate features.

        Args:
            batch_size: Size of the original batch (half of total since we concat orig + adv)

        Returns:
            FDA loss value
        """
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_layers = 0

        for _, feats in self.features.items():
            if feats.size(0) < batch_size * 2:
                continue

            # Split the concatenated batch
            x_feats = feats[:batch_size]
            adv_feats = feats[batch_size : batch_size * 2]

            # Compute mean across spatial dimensions but keep channel dimension
            if len(x_feats.shape) == 4:  # Conv features (B, C, H, W)
                mean_features = torch.mean(x_feats, dim=(2, 3), keepdim=True)
                mean_features = mean_features.expand_as(x_feats)
            elif len(x_feats.shape) == 2:  # FC features (B, C)
                mean_features = torch.mean(x_feats, dim=1, keepdim=True)
                mean_features = mean_features.expand_as(x_feats)
            else:
                continue

            # Create masks for features below and above mean
            mask_below = (x_feats < mean_features).float()
            mask_above = (x_feats >= mean_features).float()

            # Compute weighted features
            weighted_below = mask_below * adv_feats
            weighted_above = mask_above * adv_feats

            # Compute L2 norms
            norm_below = torch.norm(weighted_below) / torch.sqrt(
                torch.tensor(adv_feats.numel()).float()
            )
            norm_above = torch.norm(weighted_above) / torch.sqrt(
                torch.tensor(adv_feats.numel()).float()
            )

            # Add epsilon to prevent log(0)
            eps = 1e-8
            layer_loss = torch.log(norm_below + eps) - torch.log(norm_above + eps)
            loss = loss + layer_loss
            num_layers += 1

        if num_layers > 0:
            loss = loss / num_layers

        return loss

    def __del__(self) -> None:
        """Clean up hooks when object is destroyed."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=FDA,
        attack_args={'eps': 8 / 255, 'steps': 40},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
