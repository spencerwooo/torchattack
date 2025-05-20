from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

from torchattack._rgetattr import rgetattr
from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class BFA(Attack):
    """The BFA (Black-box Feature) attack.

    > From the paper: [Improving the transferability of adversarial examples through
    black-box feature attacks](https://www.sciencedirect.com/science/article/pii/S0925231224006349).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        eta: The mask gradient's perturbation size. Defaults to 28.
        num_ens: Number of aggregate gradients. Defaults to 30.
        feature_layer_cfg: Name of the feature layer to attack. If not provided, tries
            to infer from built-in config based on the model name. Defaults to ""
        num_classes: Number of classes. Defaults to 1000.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    _builtin_cfgs = {
        'inception_v3': 'Mixed_5b',
        'resnet50': 'layer2.3',  # (not present in the paper)
        'resnet152': 'layer2.7',
        'vgg16': 'features.15',
        'inception_v4': 'features.9',
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
        eta: int = 28,
        num_ens: int = 30,
        feature_layer_cfg: str = '',
        num_classes: int = 1000,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
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
        self.eta = eta
        self.num_ens = num_ens
        self.num_classes = num_classes
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        self.feature_maps = torch.empty(0)
        self.feature_layer_cfg = feature_layer_cfg

        self._register_hook()

    def _register_hook(self) -> None:
        def hook_fn(m: nn.Module, i: torch.Tensor, o: torch.Tensor) -> None:
            self.feature_maps = o

        feature_mod = rgetattr(self.model, self.feature_layer_cfg)
        feature_mod.register_forward_hook(hook_fn)

    def _get_maskgrad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x.requires_grad = True
        outs = self.model(self.normalize(x))
        loss = self.lossfn(outs, y)

        # Get gradient of the loss w.r.t. the masked image
        mg = torch.autograd.grad(loss, x)[0]
        mg /= torch.sum(torch.square(mg), dim=(1, 2, 3), keepdim=True).sqrt()
        return mg.detach()

    def _get_aggregate_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _ = self.model(self.normalize(x))
        x_masked = x.clone().detach()
        aggregate_grad = torch.zeros_like(self.feature_maps)

        # Targets mask
        t = f.one_hot(y.type(torch.int64), self.num_classes).float().to(self.device)

        # Get aggregate gradients over ensembles
        for _ in range(self.num_ens):
            g = self._get_maskgrad(x_masked, y)

            # Get fitted image
            x_masked = x + self.eta * g

            # Get mask gradient
            outs = self.model(self.normalize(x_masked))
            loss = torch.sum(outs * t, dim=1).mean()
            aggregate_grad += torch.autograd.grad(loss, self.feature_maps)[0]

        # Compute average gradient
        aggregate_grad /= -torch.sqrt(
            torch.sum(torch.square(aggregate_grad), dim=(1, 2, 3), keepdim=True)
        )
        return aggregate_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform BFA on a batch of images.

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

        aggregate_grad = self._get_aggregate_grad(x, y)

        # Perform BFA
        for _ in range(self.steps):
            # Compute loss
            _ = self.model(self.normalize(x + delta))
            loss = torch.sum(aggregate_grad * self.feature_maps, dim=(1, 2, 3)).mean()

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


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=BFA,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet152',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
