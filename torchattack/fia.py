from typing import Callable

import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class FIA(Attack):
    """The FIA (Feature Importance-aware) Attack.

    > From the paper: [Feature Importance-aware Transferable Adversarial
    Attacks](https://arxiv.org/abs/2107.14185).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        num_ens: Number of aggregate gradients. Defaults to 30.
        feature_layer_cfg: Name of the feature layer to attack. If not provided, tries
            to infer from built-in config based on the model name. Defaults to ""
        drop_rate: Dropout rate for random pixels. Defaults to 0.3.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    _builtin_cfgs = {
        'inception_v3': 'Mixed_5b',
        'resnet50': 'layer2.3',  # (not present in the paper)
        'resnet152': 'layer2.7',
        'vgg16': 'features.15',
        'inception_resnet_v2': 'conv2d_4a',
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
        num_ens: int = 30,
        feature_layer_cfg: str = '',
        drop_rate: float = 0.3,
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
        self.num_ens = num_ens
        self.drop_rate = drop_rate
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.feature_layer_cfg = feature_layer_cfg
        self.feature_module = rgetattr(self.model, feature_layer_cfg)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform FIA on a batch of images.

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

        hf = self.feature_module.register_forward_hook(self._forward_hook)
        hb = self.feature_module.register_full_backward_hook(self._backward_hook)

        # Gradient aggregation on ensembles
        agg_grad: torch.Tensor | float = 0.0
        for _ in range(self.num_ens):
            # Create variants of input with randomly dropped pixels
            x_dropped = self.drop_pixels(x)

            # Get model outputs and compute gradients
            outs_dropped = self.model(self.normalize(x_dropped))
            outs_dropped = torch.softmax(outs_dropped, dim=1)

            loss = sum(outs_dropped[bi][y[bi]] for bi in range(x.shape[0]))
            loss.backward()

            # Accumulate gradients
            agg_grad += self.mid_grad[0].detach()

        # for batch_i in range(x.shape[0]):
        #     agg_grad[batch_i] /= agg_grad[batch_i].norm(p=2)
        agg_grad /= torch.norm(agg_grad, p=2, dim=(1, 2, 3), keepdim=True)
        hb.remove()

        # Perform FIA
        for _ in range(self.steps):
            # Pass through the model
            _ = self.model(self.normalize(x + delta))

            # Hooks are updated during forward pass
            loss = (self.mid_output * agg_grad).sum()
            loss.backward()

            if delta.grad is None:
                continue

            # Apply momentum term
            g = self.decay * g + delta.grad / torch.mean(
                torch.abs(delta.grad), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data - self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # Zero out gradient
            delta.grad.detach_()
            delta.grad.zero_()

        hf.remove()
        return x + delta

    def drop_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop pixels from the input image.

        Args:
            x: A batch of images. Shape: (N, C, H, W).

        Returns:
            A batch of images with randomly dropped pixels.
        """

        x_dropped = torch.zeros_like(x)
        x_dropped.copy_(x).detach()
        x_dropped.requires_grad = True

        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.drop_rate))
        x_dropped = x_dropped * mask

        return x_dropped

    def _forward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
        self.mid_output = o

    def _backward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
        self.mid_grad = o


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        FIA,
        attack_args={'feature_layer_cfg': 'layer2'},
        model_name='resnet50',
        victim_model_names=['resnet18', 'vgg13', 'densenet121'],
    )
