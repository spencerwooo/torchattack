# Loading pretrained models

## The `AttackModel`

To launch any adversarial attack, you would need a model to attack.

torchattack provides a simple abstraction over both [torchvision](https://github.com/pytorch/vision) and [timm](https://github.com/huggingface/pytorch-image-models) models, to load pretrained image classification models on ImageNet.

First, import `torch`, import [`AttackModel`][torchattack.attack_model.AttackModel] from `torchattack`, and determine the device to use.

```python
import torch
from torchattack import AttackModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Pretrained models are loaded by its name

Contrary to `torchvision.models`, [`AttackModel`][torchattack.attack_model.AttackModel] loads a pretrained model by its name.

To load a ResNet-50 model for instance.

```python
model = AttackModel.from_pretrained(model_name='resnet50', device=device)
```

The [`AttackModel.from_pretrained()`][torchattack.attack_model.AttackModel.from_pretrained] method does three things under the hood:

1. It automatically loads the model from either `torchvision` (by default) or `timm` (if not found in `torchvision`).
2. It sets the model to evaluation mode by calling `model.eval()`, and moves the model to the specified device.
3. It resolves the model's `transform` and `normalize` functions associated with its pretrained weights to the [`AttackModel`][torchattack.attack_model.AttackModel] instance, also automatically.

Doing so, we not only get our pretrained model set up, but also its necessary associated, and more importantly, **==_separated_ transform and normalization functions==(1).**
{ .annotate }

1. Separating the model's normalize function from its transform is crucial for launching attacks, **as adversarial perturbation is crafted within the original image space — most often within `(0, 1)`.**

```python
transform, normalize = model.transform, model.normalize
```

## Specifying the model source

[`AttackModel`][torchattack.attack_model.AttackModel] honors an explicit model source to load from, by prepending the model name with `tv/` or `timm/`, for `torchvision` and `timm` respectively.

For instance, to load the ViT-B/16 model from `timm`.

```python
vit_b16 = AttackModel.from_pretrained(model_name='timm/vit_base_patch16_224', device=device)
```

To load the Inception-v3 model from `torchvision`.

```python
inv_v3 = AttackModel.from_pretrained(model_name='tv/inception_v3', device=device)
```

Or, explicitly specify using `timm` as the source with `from_timm=True`.

```python
pit_b = AttackModel.from_pretrained(model_name='pit_b_224', device=device, from_timm=True)
```
