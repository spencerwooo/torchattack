import os

import pytest
import torch
from PIL import Image

from torchattack.attack_model import AttackModel


@pytest.fixture(scope='session')
def device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope='session')
def image() -> Image.Image:
    image_path = os.path.join(os.path.dirname(__file__), 'image.png')
    return Image.open(image_path).convert('RGB')


@pytest.fixture()
def prepare_cnn_and_data(
    device, image
) -> tuple[AttackModel, tuple[torch.Tensor, torch.Tensor]]:
    model = AttackModel.from_pretrained('resnet50', device)

    x = model.transform(image).unsqueeze(0)
    y = torch.tensor([665])

    return model, (x, y)


@pytest.fixture()
def prepare_vit_and_data(
    device, image
) -> tuple[AttackModel, tuple[torch.Tensor, torch.Tensor]]:
    model = AttackModel.from_pretrained('vit_base_patch16_224', device, from_timm=True)

    x = model.transform(image).unsqueeze(0)
    y = torch.tensor([665])

    return model, (x, y)
