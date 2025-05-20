import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from torchattack.attack_model import AttackModel


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_attack_model_init_and_meta_resolve():
    model = DummyModel()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    am = AttackModel('dummy', model, transform, normalize)
    r = repr(am)
    assert 'AttackModel' in r
    assert 'dummy' in r

    assert am.meta.resize_size == 256
    assert am.meta.crop_size == 224

    assert am.meta.interpolation == InterpolationMode.BILINEAR
    assert am.meta.antialias is True
    assert am.meta.mean == (0.485, 0.456, 0.406)
    assert am.meta.std == (0.229, 0.224, 0.225)


def test_attack_model_forward_and_call():
    model = DummyModel()
    am = AttackModel('dummy', model, lambda x: x, lambda x: x)
    x = torch.randn(2, 10)
    out1 = am.forward(x)
    out2 = am(x)
    assert torch.allclose(out1, out2)
    assert out1.shape == (2, 5)


def test_create_relative_transform():
    model = DummyModel()
    am1_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    am2_transform = transforms.Compose(
        [
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ]
    )
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    am1 = AttackModel('m1', model, am1_transform, normalize)
    am2 = AttackModel('m2', model, am2_transform, normalize)

    rel_transform = am1.create_relative_transform(am2)
    # Should only contain Resize + MaybePIlToTensor
    assert isinstance(rel_transform.transforms[0], transforms.Resize)
    assert rel_transform.transforms[0].size == 224
    # Test with a tensor
    x = torch.rand(3, 224, 224)
    y = rel_transform(x)
    assert isinstance(y, torch.Tensor)
    # Test with a PIL image
    img = Image.fromarray((torch.rand(224, 224, 3).numpy() * 255).astype('uint8'))
    y2 = rel_transform(img)
    assert isinstance(y2, torch.Tensor)
