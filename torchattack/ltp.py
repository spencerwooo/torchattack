from typing import Any

import torch

from torchattack.attack import Attack, register_attack
from torchattack.generative._weights import GeneratorWeights, GeneratorWeightsEnum
from torchattack.generative.resnet_generator import ResNetGenerator


class LTPWeights(GeneratorWeightsEnum):
    """
    Pretrained weights for the LTP attack generator are sourced from [the original
    implementation of the LTP
    attack](https://github.com/krishnakanthnakka/Transferable_Perturbations#installation).
    """

    DENSENET121_IMAGENET = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/ltp_densenet121_1_net_g.pth'
    )
    INCEPTION_V3_IMAGENET = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/ltp_inception_v3_1_net_g.pth'
    )
    RESNET152_IMAGENET = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/ltp_resnet152_1_net_g.pth'
    )
    SQUEEZENET_IMAGENET = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/ltp_squeezenet_1_net_g.pth'
    )
    VGG16_IMAGENET = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/ltp_vgg16_1_net_g.pth'
    )
    DEFAULT = RESNET152_IMAGENET


@register_attack(category='GENERATIVE')
class LTP(Attack):
    """LTP Attack (Learning Transferable Adversarial Perturbations).

    > From the paper: [Learning Transferable Adversarial
    Perturbations](https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html).

    Args:
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 10/255.
        weights: Pretrained weights for the generator. Either import and use the enum,
            or use its name. Defaults to LTPWeights.DEFAULT.
        checkpoint_path: Path to a custom checkpoint. Defaults to None.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        eps: float = 10 / 255,
        weights: LTPWeights | str | None = LTPWeights.DEFAULT,
        checkpoint_path: str | None = None,
        inception: bool | None = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        # Generative attacks do not require specifying model and normalize
        super().__init__(model=None, normalize=None, device=device)

        self.eps = eps
        self.checkpoint_path = checkpoint_path
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.weights = LTPWeights.verify(weights)

        # Whether is inception or not (crop layer 3x300x300 to 3x299x299)
        is_inception = (
            inception
            if inception is not None
            else (self.weights.inception if self.weights is not None else False)
        )

        # Initialize the generator
        self.generator = ResNetGenerator(inception=is_inception)

        # Load the weights
        if self.checkpoint_path is not None:
            self.generator.load_state_dict(
                torch.load(self.checkpoint_path, weights_only=True)
            )
        elif self.weights is not None:
            self.generator.load_state_dict(self.weights.get_state_dict(check_hash=True))

        self.generator.eval().to(self.device)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform LTP on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        x_unrestricted = self.generator(x)
        delta = torch.clamp(x_unrestricted - x, -self.eps, self.eps)
        x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        return x_adv


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(
        attack=LTP,
        attack_args={'eps': 10 / 255, 'weights': 'DENSENET121_IMAGENET'},
        model_name='vgg16',
        victim_model_names=['resnet152'],
    )
