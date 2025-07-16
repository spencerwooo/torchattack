from typing import Callable

import torch
import torch.nn as nn

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class DANAA(Attack):
    """The DANAA (Double Adversarial Neuron Attribution) Attack.

    > From the paper: [DANAA: Towards Transferable Attacks with Double Adversarial
    Neuron Attribution](https://arxiv.org/abs/2310.10427).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 16/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        num_ens: Number of aggregate gradients. Defaults to 30.
        scale: Scale of random perturbation for non-linear path-based attribution. Defaults to 0.25.
        lr: Learning rate for non-linear path-based attribution. Defaults to 0.0025.
        feature_layer_cfg: Name of the feature layer to attack. If not provided, tries
            to infer from built-in config based on the model name. Defaults to ""
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    _builtin_cfgs = {
        'inception_v3': 'Mixed_5b',
        'resnet50': 'layer2.3',
        'resnet152': 'layer2.7',
        'resnet18': 'layer2',
        'vgg16': 'features.15',
        'inception_resnet_v2': 'conv2d_4a',
    }

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        eps: float = 16 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        num_ens: int = 30,
        scale: float = 0.25,
        lr: float = 0.0025,
        feature_layer_cfg: str = '',
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        # If `feature_layer_cfg` is not provided, try to infer used feature layer from
        # the `model_name` attribute (automatically attached during instantiation)
        if not feature_layer_cfg and isinstance(model, AttackModel):
            feature_layer_cfg = self._builtin_cfgs.get(model.model_name, 'layer2')

        # Delay initialization to avoid overriding the model's `model_name` attribute
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.num_ens = num_ens
        self.scale = scale
        self.lr = lr
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.feature_layer_cfg = feature_layer_cfg
        self.feature_module = rgetattr(self.model, feature_layer_cfg)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform DANAA on a batch of images.

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

        # Register the forward and backward hooks
        hf = self.feature_module.register_forward_hook(self._forward_hook)
        hb = self.feature_module.register_full_backward_hook(self._backward_hook)

        # Initialize the original input for Non-linear path-based attribution
        x_t = x.clone().detach().requires_grad_(True)

        # Gradient aggregation on ensembles
        agg_grad = 0
        for _ in range(self.num_ens):
            # Move along the non-linear path
            x_perturbed = x_t + torch.randn_like(x_t) * self.scale

            # Obtain the output
            logits = self.model(self.normalize(x_perturbed))

            # Calculate the loss
            loss = torch.softmax(logits, 1)[torch.arange(logits.shape[0]), y].sum()

            # Calculate the gradients w.r.t. the input
            x_grad = torch.autograd.grad(loss, x_t, retain_graph=True)[0]

            # Update the input
            x_t = x_t + self.lr * x_grad.sign()

            # Aggregate the gradients w.r.t. the intermediate features
            agg_grad += self.mid_grad[0].detach()  # type: ignore[assignment]

        # Normalize the aggregated gradients
        agg_grad = -agg_grad / torch.sqrt(
            torch.sum(agg_grad**2, dim=(1, 2, 3), keepdim=True)  # type: ignore[call-overload]
        )
        hb.remove()

        # Obtain the base features
        _ = self.model(self.normalize(x_t))
        y_base = self.mid_output.clone().detach()

        # Perform DANAA
        for _ in range(self.steps):
            # Pass through the model
            _ = self.model(self.normalize(x + delta))

            # Calculate the loss using DANAA attribution
            loss = self._get_danaa_loss(self.mid_output, y_base, agg_grad)
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

        hf.remove()
        return x + delta

    def _get_danaa_loss(
        self,
        mid_feature: torch.Tensor,
        base_feature: torch.Tensor,
        agg_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the DANAA loss based on balanced attribution.

        Args:
            mid_feature: The intermediate feature of adversarial input.
            base_feature: The intermediate feature of base input.
            agg_grad: The aggregated gradients w.r.t. the intermediate features.

        Returns:
            The DANAA loss.
        """
        gamma = 1.0
        attribution = (mid_feature - base_feature) * agg_grad
        blank = torch.zeros_like(attribution)
        positive = torch.where(attribution >= 0, attribution, blank)
        negative = torch.where(attribution < 0, attribution, blank)
        balance_attribution = positive + gamma * negative
        loss = torch.mean(balance_attribution)

        return loss

    def _forward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
        self.mid_output = o

    def _backward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
        self.mid_grad = o


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        DANAA,
        attack_args={'feature_layer_cfg': 'layer2'},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
