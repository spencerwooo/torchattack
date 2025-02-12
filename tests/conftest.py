import os
from typing import Callable

import pytest
import torch
from PIL import Image

from torchattack.attack_model import AttackModel


@pytest.fixture()
def device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture()
def data() -> Callable[
    [Callable[[Image.Image | torch.Tensor], torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor],
]:
    def _open_and_transform_image(
        transform: Callable[[Image.Image | torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(os.path.dirname(__file__), 'image.png')
        image = Image.open(image_path).convert('RGB')
        x = transform(image).unsqueeze(0)
        y = torch.tensor([665])
        return x, y

    return _open_and_transform_image


@pytest.fixture()
def resnet50_model() -> AttackModel:
    return AttackModel.from_pretrained('resnet50')


@pytest.fixture()
def vgg16_model() -> AttackModel:
    return AttackModel.from_pretrained('vgg16')


@pytest.fixture()
def vitb16_model() -> AttackModel:
    return AttackModel.from_pretrained('vit_base_patch16_224', from_timm=True)
