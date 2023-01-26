from typing import Optional

import torch
import torch.nn as nn

from torchattacks.base import Attack


class FGSM(Attack):
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Fast Gradient Sign Method (FGSM)

        From the paper 'Explaining and Harnessing Adversarial Examples'
        https://arxiv.org/abs/1412.6572

        Args:
            model: A torch.nn.Module network model.
            eps: Maximum perturbation measured by Linf. Defaults to 8/255.
            clip_min: Minimum value for clipping. Defaults to 0.0.
            clip_max: Maximum value for clipping. Defaults to 1.0.
            targeted: Targeted attack if True. Defaults to False.
            device: Device to use for tensors. Defaults to 'cuda' if available.
        """

        super().__init__(device=device)

        self.model = model
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform FGSM on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        # This is written in a way that is similar to iterative methods such as MIM.
        # The original implementation of FGSM is not written in this way.
        delta = torch.zeros_like(x, requires_grad=True)

        outs = self.model(x + delta)
        loss = self.lossfn(outs, y)

        if self.targeted:
            loss = -loss

        loss.backward()

        # If for some reason delta.grad is None, return the original image.
        if delta.grad is None:
            return x

        g_sign = delta.grad.data.sign()

        delta.data = delta.data + self.eps * g_sign
        delta.data = torch.clamp(delta.data, -self.eps, self.eps)
        delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

        return x + delta
