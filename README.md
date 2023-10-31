# 🛡 torchattack

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![lint](https://github.com/daisylab-bit/torchattack/actions/workflows/lint.yml/badge.svg)](https://github.com/daisylab-bit/torchattack/actions/workflows/lint.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/daisylab-bit/torchattack)](https://github.com/daisylab-bit/torchattack/releases/latest)

A set of adversarial attacks implemented in PyTorch. _For internal use._

```shell
# Install from github source
python -m pip install git+https://github.com/daisylab-bit/torchattack

# Install from gitee mirror
python -m pip install git+https://gitee.com/daisylab-bit/torchattack
```

## Usage

```python
from torchvision.models import resnet50
from torchvision.transforms import transforms

from torchattack import FGSM, MIFGSM

# Load a model
model = resnet50(weights='DEFAULT')

# Define transforms (you are responsible for normalizing the data if needed)
transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Initialize an attack
attack = FGSM(model, transform, eps=0.03)

# Initialize an attack with extra params
attack = MIFGSM(model, transform, eps=0.03, steps=10, decay=1.0)
```

Check out [`torchattack.utils.run_attack`](src/torchattack/utils.py) for a simple example.

## Attacks

Gradient-based attacks:

|    Name    |   $\ell_p$    | Paper                                                                                                                      | `torchattack` class    |
| :--------: | :-----------: | :------------------------------------------------------------------------------------------------------------------------- | :--------------------- |
|    FGSM    | $\ell_\infty$ | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)                                          | `torchattack.FGSM`     |
|    PGD     | $\ell_\infty$ | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                          | `torchattack.PGD`      |
|    PGD     |   $\ell_2$    | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                          | `torchattack.PGDL2`    |
|  MI-FGSM   | $\ell_\infty$ | [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)                                             | `torchattack.MIFGSM`   |
|  DI-FGSM   | $\ell_\infty$ | [Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978)                 | `torchattack.DIFGSM`   |
|  TI-FGSM   | $\ell_\infty$ | [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884) | `torchattack.TIFGSM`   |
|  NI-FGSM   | $\ell_\infty$ | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.NIFGSM`   |
| SI-NI-FGSM | $\ell_\infty$ | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.SINIFGSM` |
|  VMI-FGSM  | $\ell_\infty$ | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VMIFGSM`  |
|  VNI-FGSM  | $\ell_\infty$ | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VNIFGSM`  |
|   Admix    | $\ell_\infty$ | [Admix: Enhancing the Transferability of Adversarial Attacks](https://arxiv.org/abs/2102.00436)                            | `torchattack.Admix`    |

Others:

|   Name   |        $\ell_p$         | Paper                                                                                                   | `torchattack` class    |
| :------: | :---------------------: | :------------------------------------------------------------------------------------------------------ | :--------------------- |
| DeepFool |        $\ell_2$         | [DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks](https://arxiv.org/abs/1511.04599) | `torchattack.DeepFool` |
|  GeoDA   | $\ell_\infty$, $\ell_2$ | [GeoDA: A Geometric Framework for Black-box Adversarial Attacks](https://arxiv.org/abs/2003.06468)      | `torchattack.GeoDA`    |

## Development

```shell
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install deps with dev extras
python -m pip install -e '.[dev]'
```

## License

[MIT](LICENSE)
