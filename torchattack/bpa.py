from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Function

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class BPA(Attack):
    """The BPA (Backward Propagation) attack.

    > From the paper: [Rethinking the Backward Propagation for Adversarial
    Transferability](https://arxiv.org/abs/2306.12685).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        hook_cfg: Config used for applying hooks to the model. Supported values:
            `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
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
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Modify the model to use custom layers for BPA. Specifically, replaces the
        maxpool and ReLU layers of ResNets with custom implementations.
        """

        self.model.maxpool = MaxPool2dK3S2P1()
        for i in range(1, len(self.model.layer3)):
            self.model.layer3[i].relu = ReLUSiLU()

        for i in range(len(self.model.layer4)):
            self.model.layer4[i].relu = ReLUSiLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform BPA on a batch of images.

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

        # Perform BPA
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


class MaxPool2dK3S2P1Function(Function):
    temperature = 10.0

    @staticmethod
    def forward(ctx, i):
        with torch.no_grad():
            o = f.max_pool2d(i, 3, 2, 1)
        ctx.save_for_backward(i, o)
        return o.to(i.device)

    @staticmethod
    def backward(ctx, grad_out):
        with torch.no_grad():
            i, o = ctx.saved_tensors
            input_unfold = f.unfold(i, 3, padding=1, stride=2).reshape(
                (
                    i.shape[0],
                    i.shape[1],
                    3 * 3,
                    grad_out.shape[2] * grad_out.shape[3],
                )
            )

            output_unfold = torch.exp(
                MaxPool2dK3S2P1Function.temperature * input_unfold
            ).sum(dim=2, keepdim=True)

            grad_output_unfold = grad_out.reshape(o.shape[0], o.shape[1], 1, -1)
            grad_output_unfold = grad_output_unfold.repeat(1, 1, 9, 1)
            grad_input_unfold = (
                grad_output_unfold
                * torch.exp(MaxPool2dK3S2P1Function.temperature * input_unfold)
                / output_unfold
            )
            grad_input_unfold = grad_input_unfold.reshape(
                i.shape[0], -1, o.shape[2] * o.shape[3]
            )
            grad_input = f.fold(grad_input_unfold, i.shape[2:], 3, padding=1, stride=2)
            return grad_input.to(i.device)


class MaxPool2dK3S2P1(nn.Module):
    def __init__(self):
        super(MaxPool2dK3S2P1, self).__init__()

    def forward(self, input):
        return MaxPool2dK3S2P1Function.apply(input)


class ReLUSiLUFunction(Function):
    temperature = 1.0

    @staticmethod
    def forward(ctx, i):
        with torch.no_grad():
            o = torch.relu(i)
        ctx.save_for_backward(i)
        return o.to(i.device)

    @staticmethod
    def backward(ctx, grad_out):
        (i,) = ctx.saved_tensors
        with torch.no_grad():
            grad_in = i * torch.sigmoid(i) * (1 - torch.sigmoid(i)) + torch.sigmoid(i)
            grad_in = grad_in * grad_out * ReLUSiLUFunction.temperature
        return grad_in.to(i.device)


class ReLUSiLU(nn.Module):
    def __init__(self):
        super(ReLUSiLU, self).__init__()

    def forward(self, input):
        return ReLUSiLUFunction.apply(input)


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=BPA,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
