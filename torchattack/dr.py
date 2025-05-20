from typing import Callable

import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class DR(Attack):
    """The DR (Dispersion Reduction) attack.

    > From the paper: [Enhancing Cross-Task Black-Box Transferability of Adversarial
    Examples With Dispersion Reduction](https://arxiv.org/abs/1911.11616).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 100.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        feature_layer_cfg: Module layer name of the model to extract features from and
            apply dispersion reduction to. If not provided, tries to infer from built-in
            config based on the model name. Defaults to "".
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    # Specified in _builtin_models assume models that are loaded from,
    # or share the exact structure as, torchvision model variants.
    _builtin_cfgs = {
        'vgg16': 'features.14',  # conv3-3 for VGG-16
        'resnet50': 'layer2.3.conv3',  # conv3-4-3 for ResNet-50 (not present in the paper)
        'resnet152': 'layer2.7.conv3',  # conv3-8-3 for ResNet-152
        'inception_v3': 'Mixed_5b',  # Mixed_5b (Group A) for Inception-v3
    }

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 100,
        alpha: float | None = None,
        decay: float = 1.0,
        feature_layer_cfg: str = '',
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        # If `feature_layer_cfg` is not provided, try to infer used feature layer from
        # the `model_name` attribute (automatically attached during instantiation)
        if not feature_layer_cfg and isinstance(model, AttackModel):
            feature_layer_cfg = self._builtin_cfgs[model.model_name]

        # Delay initialization to avoid overriding the model's `model_name` attribute
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.features = torch.empty(0)
        self.feature_layer_cfg = feature_layer_cfg

        self._register_model_hooks()

    def _register_model_hooks(self) -> None:
        def hook_fn(m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
            self.features = o

        feature_mod = rgetattr(self.model, self.feature_layer_cfg)
        feature_mod.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform DR on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform PGD
        for _ in range(self.steps):
            # Compute loss
            _ = self.model(self.normalize(x + delta))
            loss = -self.features.std()

            # Compute gradient
            loss.backward()

            if delta.grad is None:
                continue

            # Update delta
            g = delta.grad.data.sign()

            delta.data = delta.data + self.alpha * g
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        return x + delta


if __name__ == '__main__':
    from torchattack.evaluate.runner import run_attack

    run_attack(DR, model_name='vgg16', victim_model_names=['resnet18'])
