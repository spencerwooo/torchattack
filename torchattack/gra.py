from typing import Callable

import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class GRA(Attack):
    """The GRA (Gradient Relevance) attack.

    > From the paper: [Boosting Adversarial Transferability via Gradient Relevance
    Attack](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Boosting_Adversarial_Transferability_via_Gradient_Relevance_Attack_ICCV_2023_paper.html).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        beta: The upper bound of the neighborhood. Defaults to 3.5.
        eta: The decay indicator factor. Defaults to 0.94.
        num_neighbors: Number of samples for estimating gradient variance. Defaults to 20.
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
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        beta: float = 3.5,
        eta: float = 0.94,
        num_neighbors: int = 20,
        decay: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay

        self.beta = beta
        self.num_neighbors = num_neighbors

        # According to the paper, eta=0.94 maintains a good balance between
        # effectiveness to normal and defended models.
        self.eta = eta
        self.radius = beta * eps

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def _get_avg_grad(
        self, x: torch.Tensor, y: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the gradient using the neighborhood of the perturbed image."""

        grad = torch.zeros_like(x)
        for _ in range(self.num_neighbors):
            xr = torch.empty_like(delta).uniform_(-self.radius, self.radius)
            outs = self.model(self.normalize(x + delta + xr))
            loss = self.lossfn(outs, y)
            grad += torch.autograd.grad(loss, delta)[0]
        return grad / self.num_neighbors

    def _get_decay_indicator(
        self,
        m: torch.Tensor,
        delta: torch.Tensor,
        cur_noise: torch.Tensor,
        prev_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Update the decay indicator based on the current and previous noise."""

        eq_m = torch.eq(cur_noise, prev_noise).float()
        di_m = torch.ones_like(delta) - eq_m
        m = m * (eq_m + di_m * self.eta)
        return m

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform GRA on a batch of images.

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

        # Initialize the decay indicator
        m = torch.full_like(delta, 1 / self.eta)

        # Perform GRA
        for _ in range(self.steps):
            # Compute loss
            outs = self.model(self.normalize(x + delta))
            loss = self.lossfn(outs, y)

            if self.targeted:
                loss = -loss

            # Compute gradient
            grad = torch.autograd.grad(loss, delta)[0]
            avg_grad = self._get_avg_grad(x, y, delta)

            # Update similarity (relevance) weighted gradient
            gradv = grad.reshape(grad.size(0), -1)
            avg_gradv = avg_grad.reshape(avg_grad.size(0), -1)
            s = torch.cosine_similarity(gradv, avg_gradv, dim=1).view(-1, 1, 1, 1)
            cur_grad = grad * s + avg_grad * (1 - s)

            # Save previous momentum
            prev_g = g.clone()

            # Apply momentum term
            g = self.decay * g + cur_grad / torch.mean(
                torch.abs(cur_grad), dim=(1, 2, 3), keepdim=True
            )

            # Update decay indicator
            m = self._get_decay_indicator(m, delta, g, prev_g)

            # Update delta
            delta.data = delta.data + self.alpha * m * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

        return x + delta


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=GRA,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
