from typing import Callable

import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class NAA(Attack):
    """The NAA (Neuron Attribution-based) Attack.

    > From the paper: [Improving Adversarial Transferability via Neuron Attribution-Based
    Attacks](https://arxiv.org/abs/2204.00008).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        num_ens: Number of aggregate gradients (NAA use `N` in the original paper
            instead of `num_ens` in FIA). Defaults to 30.
        feature_layer_name: The layer to compute feature importance. Defaults to "layer4".
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
        num_ens: int = 30,
        feature_layer_name: str = 'layer4',
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.num_ens = num_ens
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.feature_layer_name = feature_layer_name
        self.feature_module = rgetattr(self.model, feature_layer_name)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform NAA on a batch of images.

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

        # NAA's FIA-like gradient aggregation on ensembles
        # Aggregate gradients across multiple samples to estimate neuron importance
        agg_grad: torch.Tensor | float = 0.0
        for i in range(self.num_ens):
            # Create scaled variants of input
            xm = torch.zeros_like(x)
            xm = xm + x.clone().detach() * i / self.num_ens

            # Get model outputs and compute gradients
            outs = self.model(self.normalize(xm))
            outs = torch.softmax(outs, 1)

            loss = sum(outs[bi][y[bi]] for bi in range(x.shape[0]))
            loss.backward()

            # Accumulate gradients
            agg_grad += self.mid_grad[0].detach()

        # Average the gradients
        agg_grad /= self.num_ens
        hb.remove()

        # Get initial feature map
        xp = torch.zeros_like(x)  # x_prime
        self.model(self.normalize(xp))
        yp = self.mid_output.detach().clone()  # y_prime

        # Perform NAA
        for _ in range(self.steps):
            # Pass through the model
            _ = self.model(self.normalize(x + delta))

            # Calculate loss based on feature map diff weighted by neuron importance
            loss = ((self.mid_output - yp) * agg_grad).sum()
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

    def _forward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
        self.mid_output = o

    def _backward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
        self.mid_grad = o


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        NAA,
        attack_args={'feature_layer_name': 'layer2'},
        model_name='resnet50',
        victim_model_names=['resnet18', 'vgg13', 'densenet121'],
    )
