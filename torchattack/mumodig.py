from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class MuMoDIG(Attack):
    """The MuMoDIG (Multiple Monotonic Diversified Integrated Gradients) attack.

    > From the paper: [Improving Integrated Gradient-based Transferable Adversarial
    Examples by Refining the Integration Path](https://www.arxiv.org/abs/2412.18844).

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        n_trans: Rounds of random augmentation. Defaults to 6.
        n_base: Number of base paths. Defaults to 1.
        n_interpolate: Number of interpolation points. Defaults to 1.
        region_num: Number of regions for quantization. Defaults to 2.
        lambd: Weight for the quantization loss. Defaults to 0.65.
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
        n_trans: int = 6,
        n_base: int = 1,
        n_interpolate: int = 1,
        region_num: int = 2,
        lambd: float = 0.65,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, normalize, device)

        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay

        self.n_trans = n_trans
        self.n_base = n_base
        self.n_interpolate = n_interpolate
        self.lambd = lambd
        self.lbq = LBQ(region_num)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.lossfn = nn.CrossEntropyLoss()

    def _integrated_grad(
        self,
        xadv: torch.Tensor,
        y: torch.Tensor,
        x_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute integrated gradients along quantized baseline paths."""
        if x_anchor is None:
            x_anchor = xadv

        igrad = torch.zeros_like(xadv)

        # approximate the path integral over n_base * n_interpolate points
        for i in range(self.n_base):
            # quantize baseline, detach so gradients don't flow backward through quantization
            x_ref = self.lbq(x_anchor).detach()
            path = xadv - x_ref
            accumulated_grad = torch.zeros_like(xadv)

            for j in range(self.n_interpolate):
                # interpolate along the path
                x_inter = x_ref + (i + self.lambd) * path / self.n_interpolate

                # Ensure x_inter requires grad for autograd.grad
                x_inter.requires_grad_(True)

                outs = self.model(self.normalize(x_inter))
                loss = self.lossfn(outs, y)

                # only keep the graph for the very last step
                retain = i + 1 == self.n_base and j + 1 == self.n_interpolate
                jgrad = torch.autograd.grad(loss, x_inter, retain_graph=retain)[0]
                accumulated_grad += jgrad

            # accumulate the gradients
            igrad += accumulated_grad * path

        return igrad

    def _diverse_integrated_grad(
        self, xadv: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute diversified integrated gradients with augmented inputs."""

        igrad = torch.zeros_like(xadv)
        for _ in range(self.n_trans):
            aug_x = self._apply_random_augmentation(xadv)
            igrad += self._integrated_grad(xadv, y, x_anchor=aug_x)
        return igrad

    def _apply_random_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly apply an augmentation, either affine or resize + pad."""

        def random_affine(img: torch.Tensor) -> torch.Tensor:
            """Affine transforms: randomly shift, flip, or rotate the image tensor."""

            def vertical_shift(img: torch.Tensor) -> torch.Tensor:
                _, _, h, _ = img.shape
                shift = np.random.randint(0, h)
                return img.roll(shifts=(shift,), dims=(2,))

            def horizontal_shift(img: torch.Tensor) -> torch.Tensor:
                _, _, _, w = img.shape
                shift = np.random.randint(0, w)
                return img.roll(shifts=(shift,), dims=(3,))

            def vertical_flip(img: torch.Tensor) -> torch.Tensor:
                return img.flip(dims=(2,))

            def horizontal_flip(img: torch.Tensor) -> torch.Tensor:
                return img.flip(dims=(3,))

            def random_rotate(img: torch.Tensor) -> torch.Tensor:
                import kornia.augmentation as k

                rot = k.RandomRotation(degrees=45, p=1.0)
                rot_img: torch.Tensor = rot(img)
                return rot_img

            transforms = [
                vertical_shift,
                horizontal_shift,
                vertical_flip,
                horizontal_flip,
                random_rotate,
            ]
            choice = torch.randint(len(transforms), (1,)).item()
            return transforms[int(choice)](img)

        def random_resize_and_pad(img: torch.Tensor, dim: int = 245) -> torch.Tensor:
            """
            Resize to a random intermediate size (between original and `dim`),
            pad to square `dim` * `dim`, then resize back to original.
            """
            orig = img.shape[-1]
            target = torch.randint(min(orig, dim), max(orig, dim), (1,)).item()
            resized = f.interpolate(
                img, size=(target, target), mode='bilinear', align_corners=False
            )

            pad_total = int(dim - target)
            pad_top = int(torch.randint(0, pad_total, (1,)).item())
            pad_bottom = int(pad_total - pad_top)
            pad_left = int(torch.randint(0, pad_total, (1,)).item())
            pad_right = int(pad_total - pad_left)

            padded: torch.Tensor = f.pad(
                resized, [pad_left, pad_right, pad_top, pad_bottom], value=0
            )
            padded = f.interpolate(
                padded, size=(orig, orig), mode='bilinear', align_corners=False
            )
            return padded

        # Choose one augmentation at random
        transforms = [random_affine, random_resize_and_pad]
        idx = torch.randint(len(transforms), (1,)).item()
        aug_x: torch.Tensor = transforms[int(idx)](x)  # type: ignore[operator]
        return aug_x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform MuMoDIG on a batch of images.

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

        # Perform MuMoDIG
        for _ in range(self.steps):
            # Compute the loss and gradients
            base_grad = self._integrated_grad(x + delta, y)
            diverse_grad = self._diverse_integrated_grad(x + delta, y)
            gg_grad = base_grad + diverse_grad

            # Apply momentum term
            g = self.decay * g + gg_grad / torch.mean(
                torch.abs(gg_grad), dim=(1, 2, 3), keepdim=True
            )

            # Update delta
            delta.data = delta.data + self.alpha * g.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

        return x + delta


class LBQ(nn.Module):
    """Lower Bound Quantization: snap each pixel to the lower boundary of its region."""

    def __init__(self, region_num: int, is_channels_first: bool = False) -> None:
        super().__init__()

        self.region_num = region_num  # number of segments per channel
        self.is_channels_first = is_channels_first

    def get_params(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-channel min, max and number of splits."""

        c, _, _ = x.size()
        flat = x.view(c, -1)
        min_val = flat.min(dim=1).values
        max_val = flat.max(dim=1).values
        counts = torch.full((c,), self.region_num - 1, dtype=torch.int, device=x.device)
        return min_val, max_val, counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten batch+channel dims into C for vectorized ops
        if self.is_channels_first:
            x_flat = x  # shape: C x H x W
        else:
            b, c, h, w = x.shape
            x_flat = x.view(b * c, h, w)

        # get per-channel bounds and split counts
        min_val, max_val, counts = self.get_params(x_flat)

        # sample random percentiles for splits
        total_splits = counts.sum().item()
        rand_perc = torch.rand(int(total_splits), device=x.device)
        splits = rand_perc.view(-1, self.region_num - 1)

        # compute split positions: in [min_val, max_val) per channel
        splits = splits * (
            max_val.unsqueeze(1) - min_val.unsqueeze(1)
        ) + min_val.unsqueeze(1)

        # build left/right boundaries for each region
        left = torch.cat([min_val.unsqueeze(1), splits], dim=1)  # C x region_num
        right = torch.cat([splits, (max_val + 1e-6).unsqueeze(1)], dim=1)
        left = left[:, :, None, None]  # C x region_num x 1 x 1
        right = right[:, :, None, None]

        # assign each pixel to a region
        x_exp = x_flat.unsqueeze(1)  # C x 1 x H x W
        mask = (x_exp >= left) & (x_exp < right)  # C x region_num x H x W
        idx = mask.int().argmax(dim=1, keepdim=True)  # C x 1 x H x W

        # map each pixel to its region's lower bound
        quant_flat = torch.gather(left.expand_as(mask), 1, idx).squeeze(1)

        # restore original shape if needed
        if not self.is_channels_first:
            quant_flat = quant_flat.view(b, c, h, w)
        return quant_flat


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=MuMoDIG,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
    )
