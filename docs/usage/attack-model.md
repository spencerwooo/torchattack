# Loading pretrained models

```python
import torch
from torchattack import AttackModel
```

To launch any adversarial attack, you would need a model to attack.

torchattack provides a simple abstraction over both [torchvision](https://github.com/pytorch/vision) and [timm](https://github.com/huggingface/pytorch-image-models) models, to load pretrained image classification models on ImageNet.


## API

::: torchattack.attack_model.AttackModel
    options:
        heading_level: 3
