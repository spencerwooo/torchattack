import math
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as t

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


@register_attack()
class L2T(Attack):
    """The L2T (Learning to Transform) attack.

    > From the paper: [Learning to Transform Dynamically for Better Adversarial
    Transferability](https://arxiv.org/abs/2405.14077).

    Note:
        The L2T attack requires the `torchvision` package as it uses
        `torchvision.transforms` for image transformations.

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
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
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        num_scale: int = 5,
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
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform L2T on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        g = torch.zeros_like(x)
        delta = torch.zeros_like(x, requires_grad=True)
        aug_params = torch.zeros(len(AUG_OPS), requires_grad=True, device=self.device)

        ops_num = 2
        lr = 0.01

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Perform L2T
        for _ in range(self.steps):
            aug_probs = []
            losses = []

            for _ in range(self.num_scale):
                # Create a random aug search instance for the given number of ops
                rw_search = RWAugSearch(ops_num)

                # Randomly select ops based on the aug params
                ops_indices = select_op(aug_params, ops_num)
                # Compute the joint probs of the selected ops
                aug_prob = trace_prob(aug_params, ops_indices)

                # Update the aug search with the selected ops
                rw_search.ops_num = ops_num
                rw_search.ops_indices = ops_indices

                # Save the computed probs for the current scale to later update the aug params
                aug_probs.append(aug_prob)

                # Compute loss
                outs = self.model(self.normalize(rw_search(x + delta)))
                num_copies = math.floor((len(outs) + 0.01) / len(y))
                loss = self.lossfn(outs, y.repeat(num_copies))

                if self.targeted:
                    loss = -loss

                losses.append(loss)

            # Compute gradient
            loss = torch.stack(losses).mean()
            delta.grad = torch.autograd.grad(loss, delta)[0]

            # Compute gradient for augmentation params
            aug_loss = (torch.stack(aug_probs) * torch.stack(losses)).mean()
            aug_params.grad = torch.autograd.grad(aug_loss, aug_params)[0]

            # Update augmentation params
            aug_params.data = aug_params.data + lr * aug_params.grad
            aug_params.grad.detach_()
            aug_params.grad.zero_()

            # Apply momentum term and compute delta update
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


def select_op(op_params: torch.Tensor, num_ops: int) -> list[int]:
    prob = f.softmax(op_params, dim=0)
    op_ids = torch.multinomial(prob, num_ops, replacement=True).tolist()
    return op_ids


def trace_prob(op_params: torch.Tensor, op_ids: list[int]) -> torch.Tensor:
    probs = f.softmax(op_params, dim=0)
    # tp = 1
    # for idx in op_ids:
    #     tp = tp * probs[idx]
    # return tp
    return torch.prod(probs[op_ids])


class RWAugSearch:
    def __init__(self, ops_num: int) -> None:
        self.ops_num = ops_num  # total number of ops
        self.ops_indices = [0, 0]  # initialize selected ops
        self.ops = AUG_OPS  # list of predefined ops

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert len(self.ops_indices) == self.ops_num
        for idx in self.ops_indices:
            img = AUG_OPS[idx](img)  # type: ignore
        return img


def vertical_shift(x: torch.Tensor) -> torch.Tensor:
    _, _, w, _ = x.shape
    step = np.random.randint(low=0, high=w, dtype=np.int32)
    return x.roll(step, dims=2)  # type: ignore


def horizontal_shift(x: torch.Tensor) -> torch.Tensor:
    _, _, _, h = x.shape
    step = np.random.randint(low=0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)  # type: ignore


def vertical_flip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(dims=(2,))


def horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(dims=(3,))


def rotate45(x: torch.Tensor) -> torch.Tensor:
    return t.functional.rotate(img=x, angle=45)  # type: ignore


def rotate135(x: torch.Tensor) -> torch.Tensor:
    return t.functional.rotate(img=x, angle=135)  # type: ignore


def rotate90(x: torch.Tensor) -> torch.Tensor:
    return x.rot90(k=1, dims=(2, 3))


def rotate180(x: torch.Tensor) -> torch.Tensor:
    return x.rot90(k=2, dims=(2, 3))


def add_noise(x: torch.Tensor) -> torch.Tensor:
    return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


class Rotate:
    def __init__(self, angle: float, num_scale: int) -> None:
        self.num_scale = num_scale
        self.angle = angle

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                t.functional.rotate(img=x, angle=(self.angle / (2**i)))
                for i in range(self.num_scale)
            ]
        )


def rotate(angle: float, num_scale: int) -> Rotate:
    return Rotate(angle, num_scale)


class Sim:
    def __init__(self, num_copy: int) -> None:
        self.num_copy = num_copy

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x / (2**i) for i in range(self.num_copy)])


def sim(num_copy: int) -> Sim:
    return Sim(num_copy)


class Dim:
    def __init__(self, resize_rate: float = 1.1, diversity_prob: float = 0.5) -> None:
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(
            low=min(img_size, img_resize),
            high=max(img_size, img_resize),
            size=(1,),
            dtype=torch.int32,
        )
        rescaled = f.interpolate(
            x, size=[rnd, rnd], mode='bilinear', align_corners=False
        )

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        pad = [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()]
        padded = f.pad(rescaled, pad=pad, mode='constant', value=0)

        # resize the image back to img_size
        dx: torch.Tensor = f.interpolate(
            padded, size=[img_size, img_size], mode='bilinear', align_corners=False
        )
        return dx


def dim(resize_rate: float = 1.1, diversity_prob: float = 0.5) -> Dim:
    return Dim(resize_rate, diversity_prob)


class BlockShuffle:
    def __init__(self, num_block: int = 3, num_scale: int = 10) -> None:
        self.num_block = num_block
        self.num_scale = num_scale

    def get_length(self, length: int) -> tuple[int, ...]:
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x: torch.Tensor, dim: int) -> list[torch.Tensor]:
        lengths = self.get_length(x.size(dim))
        # perm = torch.randperm(self.num_block)
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def shuffle(self, x: torch.Tensor) -> torch.Tensor:
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat(
            [
                torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1])
                for x_strip in x_strips
            ],
            dim=dims[0],
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])


def blockshuffle(num_block: int = 3, num_scale: int = 10) -> BlockShuffle:
    return BlockShuffle(num_block, num_scale)


class Admix:
    def __init__(
        self, num_admix: int = 3, admix_strength: float = 0.2, num_scale: int = 3
    ) -> None:
        self.num_scale = num_scale
        self.num_admix = num_admix
        self.admix_strength = admix_strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        admix_images = torch.concat(
            [
                (x + self.admix_strength * x[torch.randperm(x.size(0))].detach())
                for _ in range(self.num_admix)
            ],
            dim=0,
        )
        return torch.concat([admix_images / (2**i) for i in range(self.num_scale)])


def admix(num_admix: int = 3, admix_strength: float = 0.2, num_scale: int = 3) -> Admix:
    return Admix(num_admix, admix_strength, num_scale)


class Ide:
    def __init__(self, dropout_prob: list[float] | None = None) -> None:
        if dropout_prob is None:
            dropout_prob = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.dropout_prob = dropout_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [nn.Dropout(p=prob)(x) * (1 - prob) for prob in self.dropout_prob]
        )


def ide(dropout_prob: list[float] | None) -> Ide:
    return Ide(dropout_prob)


class Masked:
    def __init__(self, num_block: int, num_scale: int = 5) -> None:
        self.num_block = num_block
        self.num_scale = num_scale

    def blockmask(self, x: torch.Tensor) -> torch.Tensor:
        _, _, w, h = x.shape

        if w == h:
            step = w / self.num_block
            points = [round(step * i) for i in range(self.num_block + 1)]

        x_copy = x.clone()
        x_block, y_block = (
            random.randint(0, self.num_block - 1),
            random.randint(0, self.num_block - 1),
        )
        x_copy[
            :,
            :,
            points[x_block] : points[x_block + 1],
            points[y_block] : points[y_block + 1],
        ] = 0

        return x_copy

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.blockmask(x) for _ in range(self.num_scale)])


def masked(num_block: int, num_scale: int = 5) -> Masked:
    return Masked(num_block, num_scale)


class SSM:
    def __init__(self, rho: float = 0.5, num_spectrum: int = 10) -> None:
        self.epsilon = 16 / 255
        self.rho = rho
        self.num_spectrum = num_spectrum

    def dct(self, x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        x_shape = x.shape
        n = x_shape[-1]
        x = x.contiguous().view(-1, n)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        mat_vc = torch.fft.fft(v)

        k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * n)
        mat_wr = torch.cos(k)
        mat_wi = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        mat_v = mat_vc.real * mat_wr - mat_vc.imag * mat_wi
        if norm == 'ortho':
            mat_v[:, 0] /= np.sqrt(n) * 2
            mat_v[:, 1:] /= np.sqrt(n / 2) * 2

        mat_v = 2 * mat_v.view(*x_shape)

        return mat_v  # type: ignore

    def idct(self, mat_x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        x_shape = mat_x.shape
        n = x_shape[-1]

        mat_xv = mat_x.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            mat_xv[:, 0] *= np.sqrt(n) * 2
            mat_xv[:, 1:] *= np.sqrt(n / 2) * 2

        k = (
            torch.arange(x_shape[-1], dtype=mat_x.dtype, device=mat_x.device)[None, :]
            * np.pi
            / (2 * n)
        )
        mat_wr = torch.cos(k)
        mat_wi = torch.sin(k)

        mat_vtr = mat_xv
        mat_vti = torch.cat([mat_xv[:, :1] * 0, -mat_xv.flip([1])[:, :-1]], dim=1)

        mat_vr = mat_vtr * mat_wr - mat_vti * mat_wi
        mat_vi = mat_vtr * mat_wi + mat_vti * mat_wr

        mat_v = torch.cat([mat_vr.unsqueeze(2), mat_vi.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=mat_v[:, :, 0], imag=mat_v[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, : n - (n // 2)]
        x[:, 1::2] += v.flip([1])[:, : n // 2]

        return x.view(*x_shape).real  # type: ignore

    def dct_2d(self, x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        x1 = self.dct(x, norm=norm)
        x2 = self.dct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

    def idct_2d(self, x: torch.Tensor, norm: str | None = None) -> torch.Tensor:
        x1 = self.idct(x, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_idct = []

        for _ in range(self.num_spectrum):
            gauss = torch.randn(x.size()[0], 3, 224, 224) * self.epsilon
            gauss = gauss.cuda()
            x_dct = self.dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
            x_idct.append(self.idct_2d(x_dct * mask))

        return torch.cat(x_idct)


def ssm(rho: float = 0.5, num_spectrum: int = 10) -> SSM:
    return SSM(rho, num_spectrum)


class Crop:
    def __init__(self, ratio: float, num_scale: int = 5) -> None:
        self.num_scale = num_scale
        self.ratio = ratio

    def crop(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        width = int(x.shape[2] * ratio)
        height = int(x.shape[3] * ratio)

        left = 0 + (x.shape[2] - width) // 2
        top = 0 + (x.shape[3] - height) // 2
        cx: torch.Tensor = t.functional.resized_crop(
            x, top, left, height, width, (224, 224)
        )
        return cx

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.crop(x, self.ratio + (1 - self.ratio) * (i + 1) / self.num_scale)
                for i in range(self.num_scale)
            ]
        )


def crop(ratio: float, num_scale: int = 5) -> Crop:
    return Crop(ratio, num_scale)


class Affine:
    def __init__(self, offset: float, num_scale: int = 5) -> None:
        self.num_scale = num_scale
        self.offset = offset

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                t.functional.affine(
                    img=x,
                    angle=0,
                    translate=[
                        self.offset * (i + 1) / self.num_scale,
                        self.offset * (i + 1) / self.num_scale,
                    ],
                    scale=1,
                    shear=0,
                )
                for i in range(self.num_scale)
            ]
        )


def affine(offset: float, num_scale: int = 5) -> Affine:
    return Affine(offset, num_scale)


AUG_OPS = [
    identity,  # 0
    rotate(30, 5),
    rotate(60, 5),
    rotate(90, 5),
    rotate(120, 5),
    rotate(150, 5),
    rotate(180, 5),
    rotate(210, 5),
    rotate(240, 5),
    rotate(270, 5),
    rotate(300, 5),  # 1-10
    sim(1),
    sim(2),
    sim(3),
    sim(4),
    sim(5),
    sim(6),
    sim(7),
    sim(8),
    sim(9),
    sim(10),  # 11-20
    dim(1.1),
    dim(1.15),
    dim(1.2),
    dim(1.25),
    dim(1.3),
    dim(1.35),
    dim(1.4),
    dim(1.45),
    dim(1.5),
    dim(1.55),  # 21-30
    blockshuffle(3),
    blockshuffle(4),
    blockshuffle(5),
    blockshuffle(6),
    blockshuffle(7),
    blockshuffle(8),
    blockshuffle(9),
    blockshuffle(10),
    blockshuffle(11),
    blockshuffle(12),  # 31-40
    admix(1, 0.2),
    admix(2, 0.2),
    admix(3, 0.2),
    admix(4, 0.2),
    admix(5, 0.2),
    admix(1, 0.4),
    admix(2, 0.4),
    admix(3, 0.4),
    admix(4, 0.4),
    admix(5, 0.4),  # 41-50
    ide([0.1]),
    ide([0.1, 0.2]),
    ide([0.1, 0.2, 0.3]),
    ide([0.1, 0.2, 0.3, 0.4]),
    ide([0.1, 0.2, 0.3, 0.4, 0.5]),
    ide([0.2, 0.3, 0.4, 0.5]),
    ide([0.1, 0.3, 0.4, 0.5]),
    ide([0.1, 0.2, 0.4, 0.5]),
    ide([0.1, 0.2, 0.3, 0.5]),
    ide([0.1, 0.2, 0.3, 0.4]),  # 51-60
    masked(2),
    masked(4),
    masked(6),
    masked(8),
    masked(10),
    masked(3),
    masked(5),
    masked(7),
    masked(9),
    masked(11),  # 61-70
    ssm(0.2),
    ssm(0.4),
    ssm(0.5),
    ssm(0.6),
    ssm(0.8),
    ssm(0.1),
    ssm(0.3),
    ssm(0.7),
    ssm(0.9),  # 71-80
    crop(0.1),
    crop(0.2),
    crop(0.3),
    crop(0.4),
    crop(0.5),
    crop(0.6),
    crop(0.7),
    crop(0.8),
    crop(0.9),  # 81-90
    affine(0.5),
    affine(0.55),
    affine(0.6),
    affine(0.65),
    affine(0.7),
    affine(0.75),
    affine(0.8),
    affine(0.85),
    affine(0.9),  # 91-100
]

if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=L2T,
        attack_args={'eps': 8 / 255, 'steps': 10},
        model_name='resnet18',
        victim_model_names=['resnet50', 'vgg13', 'densenet121'],
        batch_size=2,
        # save_adv_batch=6,
    )
