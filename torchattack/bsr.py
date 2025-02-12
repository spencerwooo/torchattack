import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as t

from torchattack._attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class BSR(Attack):
    """The BSR (Block Shuffle and Rotation) attack.

    > From the paper: [Boosting Adversarial Transferability by Block Shuffle and
    Rotation](https://arxiv.org/abs/2308.10299).

    Note:
        The BSR attack requires the `torchvision` package as it uses
        `torchvision.transforms` for image transformations.


    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        num_scale: Number of scaled inputs. Defaults to 20.
        num_block: Number of blocks. Defaults to 3.
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
        decay: float = 1.0,
        num_scale: int = 20,
        num_block: int = 3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay

        self.num_scale = num_scale
        self.num_block = num_block

        # Declare random rotation transform
        self.randrot = t.RandomRotation(
            degrees=(-24, 24), interpolation=t.InterpolationMode.BILINEAR
        )

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform BSR on a batch of images.

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

        # Perform BSR
        for _ in range(self.steps):
            # Compute loss
            outs = self.model(self.normalize(self._bsr_transform(x + delta)))
            loss = self.lossfn(outs, y.repeat(self.num_scale))

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

    def _gen_rand_lengths(self, length: int) -> tuple[int]:
        """Generate a tuple of random lengths that sum up to the given length. These
        lengths are used to split a tensor into blocks of varying sizes.

        Example:
            _gen_rand_lengths(10) -> (3, 3, 4)

        Args:
            length: The total length to split.

        Returns:
            The randomly generated lengths of the split.
        """

        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def _shuffle_single_dim(self, x: torch.Tensor, dim: int) -> list[torch.Tensor]:
        """Shuffles the elements of a specified dimension in a tensor.

        Uses the lengths generated by `self._gen_rand_lengths` to split a tensor along a
        specified dimension into blocks of random lengths. Blocks are then shuffled.

        Args:
            x: The input tensor.
            dim: The dimension along which to shuffle the elements.

        Returns:
            A list of shuffled tensors.
        """

        lengths = self._gen_rand_lengths(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def _bsr_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the BSR (Block Shuffle and Rotate) transformation to the input tensor.

        This method shuffles the input tensor along two specified dimensions, applies
        random rotations to the shuffled strips, and concatenates the results.

        Args:
            x: Input tensor to be shuffled and rotated.

        Returns:
            A tensor that has been shuffled by blocks and rotated randomly.
        """

        dims = [2, 3]
        random.shuffle(dims)
        d1, d2 = dims

        # Shuffle x along the first chosen dim
        x_strips = self._shuffle_single_dim(x, d1)

        # For each strip, apply random rotation and then shuffle along the second dim
        rotated_strips = []
        for x_strip in x_strips:
            rotated = self.randrot(x_strip)
            shuffled = self._shuffle_single_dim(rotated, dim=d2)
            rotated_strips.append(torch.cat(shuffled, dim=d2))

        # Concatenate the processed strips along the first dim
        return torch.cat(rotated_strips, dim=d1)

    def _bsr_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._bsr_shuffle(x) for _ in range(self.num_scale)])


if __name__ == '__main__':
    from torchattack.eval import run_attack

    run_attack(
        attack=BSR,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
        save_adv_batch=4,
    )
