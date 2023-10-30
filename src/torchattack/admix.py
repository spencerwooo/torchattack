from typing import Callable

import torch
import torch.nn as nn

from torchattack.base import Attack


class Admix(Attack):
    """The Admix attack.

    From the paper 'Admix: Enhancing the Transferability of Adversarial Attacks',
    https://arxiv.org/abs/2102.00436
    """

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        portion: float = 0.2,
        size: int = 3,
        num_classes: int = 1000,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Admix attack.

        Args:
            model: The model to attack.
            transform: A transform to normalize images.
            eps: The maximum perturbation. Defaults to 8/255.
            steps: Number of steps. Defaults to 10.
            alpha: Step size, `eps / steps` if None. Defaults to None.
            decay: Decay factor for the momentum term. Defaults to 1.0.
            portion: Portion for the mixed image. Defaults to 0.2.
            size: Number of randomly sampled images. Defaults to 3.
            num_classes: Number of classes of the dataset used. Defaults to 1001.
            clip_min: Minimum value for clipping. Defaults to 0.0.
            clip_max: Maximum value for clipping. Defaults to 1.0.
            targeted: Targeted attack if True. Defaults to False.
            device: Device to use for tensors. Defaults to cuda if available.
        """

        super().__init__(transform, device)

        self.model = model
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.portion = portion
        self.size = size
        self.num_classes = num_classes
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform Admix on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        # delta = torch.zeros_like(x, requires_grad=True)
        g = torch.zeros_like(x)
        x_adv = x.clone().detach()

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Admix + MI-FGSM
        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            # Add delta to original image then admix
            x_admix = self.admix(x_adv)
            x_admixs = [x_admix, x_admix / 2, x_admix / 4, x_admix / 8, x_admix / 16]
            x_admixs = torch.cat(x_admixs)

            # Compute loss
            outs = self.model(self.transform(x_admixs))

            # One-hot encode labels for all admixed images
            one_hot = nn.functional.one_hot(y, self.num_classes)
            one_hot = torch.cat([one_hot] * 5 * self.size).float()

            loss = self.lossfn(outs, one_hot)

            if self.targeted:
                loss = -loss

            # # Compute gradient
            # loss.backward()

            # if x_adv.grad is None:
            #     continue

            # Gradients on Admix images
            gr_splits = torch.tensor([1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]).to(x.device)
            grad = torch.autograd.grad(loss, x_admixs)[0]
            grad = torch.mean(
                torch.chunk(grad, 5, dim=0) * gr_splits[:, None, None, None, None],
                dim=0,
            )
            grad = torch.sum(torch.chunk(grad, self.size, dim=0), dim=0)

            # Apply momentum term
            g = self.decay * g + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True
            )

            # # Update delta
            # delta.data = delta.data + self.alpha * g.sign()
            # delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            # delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

            # # Zero out gradient
            # delta.grad.detach_()
            # delta.grad.zero_()
            x_adv = x_adv.detach() + self.alpha * g.sign()
            x_adv = x + torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        return x_adv

    def admix(self, x: torch.Tensor) -> torch.Tensor:
        def x_admix(x: torch.Tensor) -> torch.Tensor:
            return x + self.portion * x[torch.randperm(x.shape[0])]

        return torch.cat([(x_admix(x)) for _ in range(self.size)])


if __name__ == '__main__':
    from torchattack.utils import run_attack

    run_attack(Admix, {'eps': 8 / 255, 'steps': 10})
