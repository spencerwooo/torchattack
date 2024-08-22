# 🛡 torchattack

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![lint](https://github.com/daisylab-bit/torchattack/actions/workflows/lint.yml/badge.svg)](https://github.com/daisylab-bit/torchattack/actions/workflows/lint.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/daisylab-bit/torchattack)](https://github.com/daisylab-bit/torchattack/releases/latest)

A set of adversarial attacks in PyTorch.

```shell
# Install from github source
python -m pip install git+https://github.com/daisylab-bit/torchattack

# Install from gitee mirror
python -m pip install git+https://gitee.com/daisylab-bit/torchattack
```

## Usage

```python
import torch
from torchattack import FGSM, MIFGSM
from torchattack.eval import AttackModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a model
model = AttackModel.from_pretrained(model_name='resnet50', device=device)
transform, normalize = model.transform, model.normalize

# Initialize an attack
attack = FGSM(model, normalize, device)

# Initialize an attack with extra params
attack = MIFGSM(model, normalize, device, eps=0.03, steps=10, decay=1.0)
```

Check out [`torchattack.eval.run_attack`](src/torchattack/eval.py) for a simple example.

## Attacks

Gradient-based attacks:

|     Name     |   $\ell_p$    | Paper                                                                                                                      | `torchattack` class       |
| :----------: | :-----------: | :------------------------------------------------------------------------------------------------------------------------- | :------------------------ |
|     FGSM     | $\ell_\infty$ | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)                                          | `torchattack.FGSM`        |
|     PGD      | $\ell_\infty$ | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                          | `torchattack.PGD`         |
|   PGD (L2)   |   $\ell_2$    | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                          | `torchattack.PGDL2`       |
|   MI-FGSM    | $\ell_\infty$ | [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)                                             | `torchattack.MIFGSM`      |
|   DI-FGSM    | $\ell_\infty$ | [Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978)                 | `torchattack.DIFGSM`      |
|   TI-FGSM    | $\ell_\infty$ | [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884) | `torchattack.TIFGSM`      |
|   NI-FGSM    | $\ell_\infty$ | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.NIFGSM`      |
|  SI-NI-FGSM  | $\ell_\infty$ | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)             | `torchattack.SINIFGSM`    |
|   VMI-FGSM   | $\ell_\infty$ | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VMIFGSM`     |
|   VNI-FGSM   | $\ell_\infty$ | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)           | `torchattack.VNIFGSM`     |
|    Admix     | $\ell_\infty$ | [Admix: Enhancing the Transferability of Adversarial Attacks](https://arxiv.org/abs/2102.00436)                            | `torchattack.Admix`       |
|     FIA      | $\ell_\infty$ | [Feature Importance-aware Transferable Adversarial Attacks](https://arxiv.org/abs/2107.14185)                              | `torchattack.FIA`         |
| PNA-PatchOut | $\ell_\infty$ | [Towards Transferable Adversarial Attacks on Vision Transformers](https://arxiv.org/abs/2109.04176)                        | `torchattack.PNAPatchOut` |

Others:

|   Name   |        $\ell_p$         | Paper                                                                                                   | `torchattack` class    |
| :------: | :---------------------: | :------------------------------------------------------------------------------------------------------ | :--------------------- |
| DeepFool |        $\ell_2$         | [DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks](https://arxiv.org/abs/1511.04599) | `torchattack.DeepFool` |
|  GeoDA   | $\ell_\infty$, $\ell_2$ | [GeoDA: A Geometric Framework for Black-box Adversarial Attacks](https://arxiv.org/abs/2003.06468)      | `torchattack.GeoDA`    |
|   SSP    |      $\ell_\infty$      | [A Self-supervised Approach for Adversarial Robustness](https://arxiv.org/abs/2006.04924)               | `torchattack.SSP`      |

## Development

```shell
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install deps with dev extras
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
```

## License

[MIT](LICENSE)

## Related

- [Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
