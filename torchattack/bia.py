from typing import Any

import torch

from torchattack.attack import Attack, register_attack
from torchattack.generative._weights import GeneratorWeights, GeneratorWeightsEnum
from torchattack.generative.resnet_generator import ResNetGenerator


class BIAWeights(GeneratorWeightsEnum):
    """
    Pretrained weights for the BIA attack generator are sourced from [the original
    implementation of the BIA
    attack](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack#pretrained-generators).
    RN stands for Random Normalization, and DA stands for Domain-Agnostic.
    """

    RESNET152 = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_resnet152_0.pth',
    )
    RESNET152_RN = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_resnet152_rn_0.pth',
    )
    RESNET152_DA = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_resnet152_da_0.pth',
    )
    DENSENET169 = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_densenet169_0.pth',
    )
    DENSENET169_RN = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_densenet169_rn_0.pth',
    )
    DENSENET169_DA = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_densenet169_da_0.pth',
    )
    VGG16 = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg16_0.pth',
    )
    VGG16_RN = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg16_rn_0.pth',
    )
    VGG16_DA = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg16_da_0.pth',
    )
    VGG19 = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg19_0.pth',
    )
    VGG19_RN = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg19_rn_0.pth',
    )
    VGG19_DA = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg19_da_0.pth',
    )
    DEFAULT = RESNET152_DA


@register_attack(category='GENERATIVE')
class BIA(Attack):
    """Beyond ImageNet Attack (BIA).

    > From the paper: [Beyond ImageNet Attack: Towards Crafting Adversarial Examples for
    Black-box Domains](https://arxiv.org/abs/2201.11528).

    Args:
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 10/255.
        weights: Pretrained weights for the generator. Either import and use the enum,
            or use its name. Defaults to BIAWeights.DEFAULT.
        checkpoint_path: Path to a custom checkpoint. Defaults to None.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        eps: float = 10 / 255,
        weights: BIAWeights | str | None = BIAWeights.DEFAULT,
        checkpoint_path: str | None = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        # Generative attacks do not require specifying model and normalize
        super().__init__(model=None, normalize=None, device=device)

        self.eps = eps
        self.checkpoint_path = checkpoint_path
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Initialize the generator and its weights
        self.generator = ResNetGenerator()

        # Prioritize checkpoint path over provided weights enum
        if self.checkpoint_path is not None:
            self.generator.load_state_dict(
                torch.load(self.checkpoint_path, weights_only=True)
            )
        else:
            # Verify and load weights from enum if checkpoint path is not provided
            self.weights = BIAWeights.verify(weights)
            if self.weights is not None:
                self.generator.load_state_dict(
                    self.weights.get_state_dict(check_hash=True)
                )

        self.generator.eval().to(self.device)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform BIA on a batch of images.

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
        attack=BIA,
        attack_args={'eps': 10 / 255, 'weights': 'VGG16_DA'},
        model_name='vgg16',
        victim_model_names=['resnet152'],
    )
