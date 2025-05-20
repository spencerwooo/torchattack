from typing import Callable

import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class ILPD(Attack):
    """The ILPD (Intermediate-level Perturbation Decay) Attack.

    > From the paper: [Improving Adversarial Transferability via Intermediate-level
    Perturbation Decay](https://arxiv.org/abs/2304.13410).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        sigma: Standard deviation for noise. Defaults to 0.05.
        feature_layer_cfg: Name of the feature layer to attack. If not provided, tries
            to infer from built-in config based on the model name. Defaults to ""
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    _builtin_cfgs = {
        'vgg19': 'features.27',
        'resnet50': 'layer2.3',
    }

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        sigma: float = 0.05,
        coef: float = 0.1,
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
        self.sigma = sigma
        self.coef = coef
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.lossfn = nn.CrossEntropyLoss()

        self.feature_layer_cfg = feature_layer_cfg
        self.feature_module = rgetattr(self.model, feature_layer_cfg)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform ILPD on a batch of images.

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

        # Perform ILPD
        for _ in range(self.steps):
            with torch.no_grad():
                ilh = self.feature_module.register_forward_hook(self._il_hook)

                xsig = x + self.sigma * torch.randn_like(x)
                self.model(self.normalize(xsig))

                ilo = self.feature_module.output
                ilh.remove()

            pdh = self._get_hook_pd(ilo, self.coef)
            self.hook = self.feature_module.register_forward_hook(pdh)

            # Pass through the model
            outs = self.model(self.normalize(x + delta))
            loss = self.lossfn(outs, y)

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

            # Clean up hooks
            self.hook.remove()

        return x + delta

    def _il_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
        """Intermediate-level hook function."""
        m.output = o

    @staticmethod
    def _get_hook_pd(oo: torch.Tensor, gamma: float) -> Callable:
        """Get the hook function for perturbation decay.

        Args:
            oo: The original output tensor of the module.
            gamma: The decay factor.

        Returns:
            The hook function.
        """

        def hook_pd(m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            return gamma * o + (1 - gamma) * oo

        return hook_pd


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        ILPD,
        attack_args={'feature_layer_cfg': 'layer2.3'},
        model_name='resnet50',
        victim_model_names=['resnet18', 'vgg13', 'densenet121'],
    )
