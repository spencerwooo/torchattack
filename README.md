<div align="center">
  <div><img src="https://github.com/user-attachments/assets/6e94a09e-557e-4705-a80d-f1ca90a23421" alt="torchattack banner" width="640" /></div>

  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
  </a>
  <a href="https://github.com/spencerwooo/torchattack/actions/workflows/ci.yml">
    <img src="https://github.com/spencerwooo/torchattack/actions/workflows/ci.yml/badge.svg" alt="lint">
  </a>
  <a href="https://github.com/spencerwooo/torchattack/releases/latest">
    <img src="https://img.shields.io/github/v/release/spencerwooo/torchattack" alt="GitHub release (latest by date)">
  </a>
</div>

---

🛡 **torchattack** - A set of adversarial attacks in PyTorch.

<sub><b>Install from GitHub source -</b></sub>

```shell
python -m pip install git+https://github.com/spencerwooo/torchattack
```

<sub><b>Install from Gitee mirror -</b></sub>

```shell
python -m pip install git+https://gitee.com/spencerwoo/torchattack
```

## Usage

```python
import torch

from torchattack import AttackModel, FGSM, MIFGSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a model
model = AttackModel.from_pretrained(model_name='resnet50', device=device)
transform, normalize = model.transform, model.normalize

# Initialize an attack
attack = FGSM(model, normalize, device)

# Initialize an attack with extra params
attack = MIFGSM(model, normalize, device, eps=0.03, steps=10, decay=1.0)
```

Check out [`torchattack.eval.runner`](torchattack/eval/runner.py) for a quick example.

## Attacks

Gradient-based attacks:

|     Name     |   $\ell_p$    | Publication | Paper (Open Access)                                                                                                                                      | Class Name    |
| :----------: | :-----------: | :---------: | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
|     FGSM     | $\ell_\infty$ |  ICLR 2015  | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)                                                                        | `FGSM`        |
|     PGD      | $\ell_\infty$ |  ICLR 2018  | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                                                        | `PGD`         |
|   PGD (L2)   |   $\ell_2$    |  ICLR 2018  | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)                                                        | `PGDL2`       |
|   MI-FGSM    | $\ell_\infty$ |  CVPR 2018  | [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)                                                                           | `MIFGSM`      |
|   DI-FGSM    | $\ell_\infty$ |  CVPR 2019  | [Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978)                                               | `DIFGSM`      |
|   TI-FGSM    | $\ell_\infty$ |  CVPR 2019  | [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884)                               | `TIFGSM`      |
|   NI-FGSM    | $\ell_\infty$ |  ICLR 2020  | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)                                           | `NIFGSM`      |
|  SI-NI-FGSM  | $\ell_\infty$ |  ICLR 2020  | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281)                                           | `SINIFGSM`    |
|   VMI-FGSM   | $\ell_\infty$ |  CVPR 2021  | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)                                         | `VMIFGSM`     |
|   VNI-FGSM   | $\ell_\infty$ |  CVPR 2021  | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)                                         | `VNIFGSM`     |
|    Admix     | $\ell_\infty$ |  ICCV 2021  | [Admix: Enhancing the Transferability of Adversarial Attacks](https://arxiv.org/abs/2102.00436)                                                          | `Admix`       |
|     FIA      | $\ell_\infty$ |  ICCV 2021  | [Feature Importance-aware Transferable Adversarial Attacks](https://arxiv.org/abs/2107.14185)                                                            | `FIA`         |
| PNA-PatchOut | $\ell_\infty$ |  AAAI 2022  | [Towards Transferable Adversarial Attacks on Vision Transformers](https://arxiv.org/abs/2109.04176)                                                      | `PNAPatchOut` |
|     SSA      | $\ell_\infty$ |  ECCV 2022  | [Frequency Domain Model Augmentation for Adversarial Attack](https://arxiv.org/abs/2207.05382)                                                           | `SSA`         |
|     TGR      | $\ell_\infty$ |  CVPR 2023  | [Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization](https://arxiv.org/abs/2303.15754)                           | `TGR`         |
|    DeCoWA    | $\ell_\infty$ |  AAAI 2024  | [Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping](https://arxiv.org/abs/2402.03951)                           | `DeCoWA`      |
|     VDC      | $\ell_\infty$ |  AAAI 2024  | [Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection](https://ojs.aaai.org/index.php/AAAI/article/view/28541) | `VDC`         |

Others:

|   Name   |        $\ell_p$         | Publication | Paper (Open Access)                                                                                     | Class Name |
| :------: | :---------------------: | :---------: | ------------------------------------------------------------------------------------------------------- | ---------- |
| DeepFool |        $\ell_2$         |  CVPR 2016  | [DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks](https://arxiv.org/abs/1511.04599) | `DeepFool` |
|  GeoDA   | $\ell_\infty$, $\ell_2$ |  CVPR 2020  | [GeoDA: A Geometric Framework for Black-box Adversarial Attacks](https://arxiv.org/abs/2003.06468)      | `GeoDA`    |
|   SSP    |      $\ell_\infty$      |  CVPR 2020  | [A Self-supervised Approach for Adversarial Robustness](https://arxiv.org/abs/2006.04924)               | `SSP`      |

## Development

```shell
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install deps with dev extras
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

## License

[MIT](LICENSE)

## Related

- [Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
