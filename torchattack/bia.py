import torch

from torchattack.generative._inference import GenerativeAttack
from torchattack.generative._weights import Weights, WeightsEnum
from torchattack.generative.resnet_generator import ResNetGenerator


class BIAWeights(WeightsEnum):
    RESNET152 = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_resnet152_0.pth',
    )
    RESNET152_RN = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_resnet152_rn_0.pth',
    )
    RESNET152_DA = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_resnet152_da_0.pth',
    )
    DENSENET169 = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_densenet169_0.pth',
    )
    DENSENET169_RN = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_densenet169_rn_0.pth',
    )
    DENSENET169_DA = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_densenet169_da_0.pth',
    )
    VGG16 = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg16_0.pth',
    )
    VGG16_RN = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg16_rn_0.pth',
    )
    VGG16_DA = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg16_da_0.pth',
    )
    VGG19 = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg19_0.pth',
    )
    VGG19_RN = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg19_rn_0.pth',
    )
    VGG19_DA = Weights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/bia_vgg19_da_0.pth',
    )
    DEFAULT = RESNET152_DA


class BIA(GenerativeAttack):
    """Beyond ImageNet Attack (BIA).

    From the paper 'Beyond ImageNet Attack: Towards Crafting Adversarial Examples for
    Black-box Domains', https://arxiv.org/abs/2201.11528

    Args:
        device: Device to use for tensors. Defaults to cuda if available. eps: The
        maximum perturbation. Defaults to 10/255. weights: Pretrained weights for the
        generator. Either import and use the enum,
            or use its name. Defaults to BIAWeights.DEFAULT.
        checkpoint_path: Path to a custom checkpoint. Defaults to None. clip_min:
        Minimum value for clipping. Defaults to 0.0. clip_max: Maximum value for
        clipping. Defaults to 1.0.
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
        super().__init__(device, eps, weights, checkpoint_path, clip_min, clip_max)

    def _init_generator(self) -> ResNetGenerator:
        generator = ResNetGenerator()
        # Prioritize checkpoint path over provided weights enum
        if self.checkpoint_path is not None:
            generator.load_state_dict(torch.load(self.checkpoint_path))
        else:
            # Verify and load weights from enum if checkpoint path is not provided
            self.weights: BIAWeights = BIAWeights.verify(self.weights)
            if self.weights is not None:
                generator.load_state_dict(self.weights.get_state_dict(check_hash=True))
        return generator.eval().to(self.device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack = BIA(device, weights='VGG19_DA')
    print(attack)
