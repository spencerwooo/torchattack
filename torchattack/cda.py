import torch

from torchattack.generative._inference import GenerativeAttack
from torchattack.generative._weights import Weights, WeightsEnum
from torchattack.generative.resnet_generator import ResNetGenerator


class CDAWeights(WeightsEnum):
    RESNET152_IMAGENET1K = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_res152_imagenet_0_rl.pth',
    )
    INCEPTION_V3_IMAGENET1K = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_incv3_imagenet_0_rl.pth',
    )
    VGG16_IMAGENET1K = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_vgg16_imagenet_0_rl.pth',
    )
    VGG19_IMAGENET1K = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/cda_vgg19_imagenet_0_rl.pth',
    )
    DEFAULT = RESNET152_IMAGENET1K


class CDA(GenerativeAttack):
    """Cross-domain Attack (CDA).

    From the paper 'Cross-Domain Transferability of Adversarial Perturbations',
    https://arxiv.org/abs/1905.11736

    Args:
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 10/255.
        weights: Pretrained weights for the generator. Defaults to CDAWeights.DEFAULT.
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
        super().__init__(device, eps, weights, checkpoint_path, clip_min, clip_max)

    def _init_generator(self) -> ResNetGenerator:
        generator = ResNetGenerator()
        # Prioritize checkpoint path over provided weights enum
        if self.checkpoint_path is not None:
            generator.load_state_dict(torch.load(self.checkpoint_path))
        else:
            # Verify and load weights from enum if checkpoint path is not provided
            self.weights: CDAWeights = CDAWeights.verify(self.weights)
            if self.weights is not None:
                generator.load_state_dict(self.weights.get_state_dict(check_hash=True))
        return generator.eval().to(self.device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack = CDA(device, eps=8 / 255, weights='VGG19_IMAGENET1K')
    print(attack)
