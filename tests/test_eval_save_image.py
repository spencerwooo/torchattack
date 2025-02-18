import numpy as np
import pytest
import torch
from PIL import Image

from torchattack.evaluate import save_image_batch


@pytest.fixture()
def images_and_names():
    imgs = torch.rand(3, 3, 224, 224)
    img_names = ['nggyu', 'nglyd', 'nglyu']
    return imgs, img_names


def test_save_image_batch_png_lossless(images_and_names, tmp_path):
    imgs, img_names = images_and_names
    save_image_batch(imgs, tmp_path, img_names, extension='png')

    for img, name in zip(imgs, img_names):
        saved_img = Image.open(tmp_path / f'{name}.png').convert('RGB')
        saved_img = np.array(saved_img, dtype=np.uint8) / 255.0
        saved_img = torch.from_numpy(saved_img).permute(2, 0, 1).contiguous().float()

        assert saved_img.size() == img.size()
        assert torch.allclose(img, saved_img, atol=4e-3)  # eps = 1/255


def test_save_image_batch_jpg_lossless(images_and_names, tmp_path):
    imgs, img_names = images_and_names
    save_image_batch(imgs, tmp_path, img_names, extension='jpeg')

    for img, name in zip(imgs, img_names):
        saved_img = Image.open(tmp_path / f'{name}.jpeg').convert('RGB')
        saved_img = np.array(saved_img, dtype=np.uint8) / 255.0
        saved_img = torch.from_numpy(saved_img).permute(2, 0, 1).contiguous().float()

        assert saved_img.size() == img.size()
        assert torch.allclose(img, saved_img, atol=8e-3)  # eps = 2/255
