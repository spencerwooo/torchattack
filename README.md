# 🛡 torchattack

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A set of adversarial attacks implemented in PyTorch. _For internal use._

```shell
pip install git+https://github.com/daisylab-bit/torchattack
```

## Usage

```python
from torchattack import FGSM, MIFGSM

# Initialize an attack
attack = FGSM(model, eps=0.03)

# Initialize an attack with extra params
attack = MIFGSM(model, eps=0.03, steps=10, decay=1.0)
```

## Attacks

| Name       | Paper                                                                                                                      | `torchattack` class    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| FGSM       | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)                                          | `torchattack.FGSM`     |
| PGD        | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                          | `torchattack.PGD`      |
| MI-FGSM    | [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)                                             | `torchattack.MIFGSM`   |
| DI-FGSM    | [Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978)                 | `torchattack.DIFGSM`   |
| TI-FGSM    | [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884) | `torchattack.TIFGSM`   |
| NI-FGSM    | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.NIFGSM`   |
| SI-NI-FGSM | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.SINIFGSM` |
| VMI-FGSM   | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VMIFGSM`  |
| VNI-FGSM   | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VNIFGSM`  |

## License

[MIT](LICENSE)
