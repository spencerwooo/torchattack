from typing import Any

import torch

from torchattack.attack import Attack, register_attack
from torchattack.generative._weights import GeneratorWeights, GeneratorWeightsEnum
from torchattack.generative.leaky_relu_resnet_generator import ResNetGenerator


class GAMAWeights(GeneratorWeightsEnum):
    """
    We provide pretrained weights of the GAMA attack generator with training steps
    identical to the described settings in the paper and appendix. Specifically, we use
    ViT-B/16 as the backend of CLIP. Training epochs are set to 5 and 10 for the COCO
    and VOC datasets, respectively.
    """

    DENSENET169_COCO = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_dense169_coco_w_vitb16_epoch4.pth'
    )
    DENSENET169_VOC = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_dense169_voc_w_vitb16_epoch9.pth'
    )
    RESNET152_COCO = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_res152_coco_w_vitb16_epoch4.pth'
    )
    RESNET152_VOC = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_res152_voc_w_vitb16_epoch9.pth'
    )
    VGG16_COCO = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_vgg16_coco_w_vitb16_epoch4.pth'
    )
    VGG16_VOC = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_vgg16_voc_w_vitb16_epoch9.pth'
    )
    VGG19_COCO = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_vgg19_coco_w_vitb16_epoch4.pth'
    )
    VGG19_VOC = GeneratorWeights(
        url='https://github.com/spencerwooo/torchattack/releases/download/v1.0-weights/gama_vgg19_voc_w_vitb16_epoch9.pth'
    )
    DEFAULT = VGG19_COCO


@register_attack(category='GENERATIVE')
class GAMA(Attack):
    """GAMA - Generative Adversarial Multi-Object Scene Attacks.

    > From the paper: [GAMA: Generative Adversarial Multi-Object Scene
    Attacks](https://arxiv.org/abs/2209.09502).

    Args:
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 10/255.
        weights: Pretrained weights for the generator. Either import and use the enum,
            or use its name. Defaults to GAMAWeights.DEFAULT.
        checkpoint_path: Path to a custom checkpoint. Defaults to None.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        eps: float = 10 / 255,
        weights: GAMAWeights | str | None = GAMAWeights.DEFAULT,
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
            self.weights = GAMAWeights.verify(weights)
            if self.weights is not None:
                self.generator.load_state_dict(
                    self.weights.get_state_dict(check_hash=True)
                )

        self.generator.eval().to(self.device)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform GAMA on a batch of images.

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
        attack=GAMA,
        attack_args={'eps': 10 / 255, 'weights': 'DENSENET169_COCO'},
        model_name='densenet169',
        victim_model_names=['resnet50'],
    )
