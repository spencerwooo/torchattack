# 🛡 torchattack

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![lint](https://github.com/daisylab-bit/torchattack/actions/workflows/lint.yml/badge.svg)](https://github.com/daisylab-bit/torchattack/actions/workflows/lint.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/daisylab-bit/torchattack)](https://github.com/daisylab-bit/torchattack/releases/latest)

A set of adversarial attacks implemented in PyTorch. _For internal use._

```shell
python -m pip install git+https://github.com/daisylab-bit/torchattack
```

## Usage

```python
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import transforms

from torchattack import FGSM, MIFGSM

# Load a model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Define transforms (you are responsible for normalizing the data if needed)
transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Initialize an attack
attack = FGSM(model, transform, eps=0.03)

# Initialize an attack with extra params
attack = MIFGSM(model, transform, eps=0.03, steps=10, decay=1.0)
```

## Attacks

| Name       | Paper                                                                                                                      | `torchattack` class    |
| :--------- | :------------------------------------------------------------------------------------------------------------------------- | :--------------------- |
| FGSM       | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)                                          | `torchattack.FGSM`     |
| PGD        | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                          | `torchattack.PGD`      |
| PGD (L2)   | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                          | `torchattack.PGDL2`    |
| DeepFool   | [DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks](https://arxiv.org/abs/1511.04599)                    | `torchattack.DeepFool` |
| MI-FGSM    | [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)                                             | `torchattack.MIFGSM`   |
| DI-FGSM    | [Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978)                 | `torchattack.DIFGSM`   |
| TI-FGSM    | [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884) | `torchattack.TIFGSM`   |
| NI-FGSM    | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.NIFGSM`   |
| SI-NI-FGSM | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.SINIFGSM` |
| VMI-FGSM   | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VMIFGSM`  |
| VNI-FGSM   | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VNIFGSM`  |

## License

[MIT](LICENSE)
