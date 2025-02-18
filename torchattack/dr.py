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
        model_name: The name of the model to attack. Defaults to "".
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 100.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        feature_layer_name: Module layer name of the model to extract features from and
            apply dispersion reduction to. If not provided, tries to infer from built-in
            config based on `model_name`. Defaults to "".
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    # Specified in _builtin_models assume models that are loaded from,
    # or share the exact structure as, torchvision model variants.
    _builtin_models = {
        'vgg16': 'features.14',  # conv3-3 for VGG-16
        'resnet152': 'layer2.7.conv3',  # conv3-8-3 for ResNet-152
        'inception_v3': 'Mixed_5b',  # Mixed_5b (Group A) for Inception-v3
    }

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        model_name: str | None = None,
        eps: float = 8 / 255,
        steps: int = 100,
        alpha: float | None = None,
        decay: float = 1.0,
        feature_layer_name: str | None = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, normalize, device)

        # If model is initialized via `torchattack.AttackModel`, infer its model_name
        # from automatically attached attribute during instantiation.
        if not model_name and isinstance(model, AttackModel):
            model_name = model.model_name

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.clip_min = clip_min
        self.clip_max = clip_max

        if feature_layer_name:
            # Explicit feature_layer takes precedence over built-in config
            self.feature_layer_name = feature_layer_name
        elif model_name:
            # If model_name is provided, try built-in config
            self.feature_layer_name = self._builtin_models[model_name]
        else:
            raise ValueError('argument `feature_layer` must be explicitly provided.')

        # Register hooks
        self.features: torch.Tensor | None = None
        self._register_model_hooks()

    def _register_model_hooks(self) -> None:
        def hook_fn(mod: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.features = output

        module = rgetattr(self.model, self.feature_layer_name)
        module.register_forward_hook(hook_fn)

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
            self.model(self.normalize(x + delta))

            if self.features is None:
                continue

            loss = -1 * self.features.std()

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
