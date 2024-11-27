from typing import Any

import torch

from torchattack._attack import Attack
from torchattack.generative._weights import GeneratorWeights, GeneratorWeightsEnum
from torchattack.generative.resnet_generator import ResNetGenerator


class CDAWeights(GeneratorWeightsEnum):
    RESNET152_IMAGENET1K = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_res152_imagenet_0_rl.pth',
    )
    INCEPTION_V3_IMAGENET1K = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_incv3_imagenet_0_rl.pth',
    )
    VGG16_IMAGENET1K = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_vgg16_imagenet_0_rl.pth',
    )
    VGG19_IMAGENET1K = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_vgg19_imagenet_0_rl.pth',
    )
    DEFAULT = RESNET152_IMAGENET1K


class CDA(Attack):
    """Cross-domain Attack (CDA).

    From the paper 'Cross-Domain Transferability of Adversarial Perturbations',
    https://arxiv.org/abs/1905.11736

    Args:
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 10/255.
        weights: Pretrained weights for the generator. Either import and use the enum,
            or use its name. Defaults to CDAWeights.DEFAULT.
        checkpoint_path: Path to a custom checkpoint. Defaults to None.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        eps: float = 10 / 255,
        weights: CDAWeights | str | None = CDAWeights.DEFAULT,
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
            self.generator.load_state_dict(torch.load(self.checkpoint_path))
        else:
            # Verify and load weights from enum if checkpoint path is not provided
            self.weights: CDAWeights = CDAWeights.verify(weights)
            if self.weights is not None:
                self.generator.load_state_dict(
                    self.weights.get_state_dict(check_hash=True)
                )

        self.generator.eval().to(self.device)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform CDA on a batch of images.

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack = CDA(device, weights='VGG19_IMAGENET1K')
    print(attack)
