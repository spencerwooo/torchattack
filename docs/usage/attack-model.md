# Loading pretrained models

## The `AttackModel`

To launch any adversarial attack, you would need a model to attack.

torchattack provides a simple abstraction over both [torchvision](https://github.com/pytorch/vision) and [timm](https://github.com/huggingface/pytorch-image-models) models, to load pretrained image classification models on ImageNet.

First, import `torch`, import `AttackModel` from `torchattack`, and determine the device to use.

```python
import torch
from torchattack import AttackModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Pretrained models are loaded by its name

Contrary to `torchvision.models`, `AttackModel` loads a pretrained model by its name.

To load a ResNet-50 model for instance.

```python
model = AttackModel.from_pretrained(model_name='resnet50', device=device)
```

The [`AttackModel.from_pretrained()`][torchattack.attack_model.AttackModel.from_pretrained] method does three things under the hood:

1. It automatically loads the model from either `torchvision` (by default) or `timm` (if not found in `torchvision`).
2. It sets the model to evaluation mode by calling `model.eval()`, and moves the model to the specified device.
3. It resolves the model's `transform` and `normalize` functions associated with its pretrained weights to the `AttackModel` instance, also automatically.

Doing so, we not only get the pretrained model set up, but also **its necessary associated, and more importantly, ==separated== transform and normalization functions.**

```python
transform, normalize = model.transform, model.normalize
```

<!-- Load a pretrained model to attack from either torchvision or timm.

```python
from torchattack import AttackModel

# Load a model with `AttackModel`
model = AttackModel.from_pretrained(model_name='resnet50', device=device)
# `AttackModel` automatically attach the model's `transform` and `normalize` functions
transform, normalize = model.transform, model.normalize

# Additionally, to explicitly specify where to load the pretrained model from (timm or torchvision),
# prepend the model name with 'timm/' or 'tv/' respectively, or use the `from_timm` argument, e.g.
vit_b16 = AttackModel.from_pretrained(model_name='timm/vit_base_patch16_224', device=device)
inv_v3 = AttackModel.from_pretrained(model_name='tv/inception_v3', device=device)
pit_b = AttackModel.from_pretrained(model_name='pit_b_224', device=device, from_timm=True)
``` -->

## API

::: torchattack.attack_model.AttackModel
    options:
        heading_level: 3
