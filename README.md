<div align="center">
  <p><br><img src="docs/images/torchattack.png" alt="torchattack banner" width="600" /></p>
</div>

---

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/refs/heads/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pypi python versions](https://img.shields.io/pypi/pyversions/torchattack.svg?logo=pypi&logoColor=white&labelColor=2D3339)](https://pypi.python.org/pypi/torchattack)
[![pypi version](https://img.shields.io/pypi/v/torchattack.svg?logo=pypi&logoColor=white&labelColor=2D3339)](https://pypi.python.org/pypi/torchattack)
[![pypi weekly downloads](https://img.shields.io/pypi/dm/torchattack?logo=pypi&logoColor=white&labelColor=2D3339)](https://pypi.python.org/pypi/torchattack)
[![lint](https://github.com/spencerwooo/torchattack/actions/workflows/ci.yml/badge.svg)](https://github.com/spencerwooo/torchattack/actions/workflows/ci.yml)

🛡 **torchattack** - _A curated list of adversarial attacks in PyTorch, with a focus on transferable black-box attacks._

```shell
pip install torchattack
```

## Highlights

- 🛡️ A curated collection of adversarial attacks implemented in PyTorch.
- 🔍 Focuses on gradient-based transferable black-box attacks.
- 📦 Easily load pretrained models from torchvision or timm using `AttackModel`.
- 🔄 Simple interface to initialize attacks with `create_attack`.
- 🔧 Extensively typed for better code quality and safety.
- 📊 Tooling for fooling rate metrics and model evaluation in `eval`.
- 🔁 Numerous attacks reimplemented for readability and efficiency (TGR, VDC, etc.).

## Documentation

torchattack's docs are available at [docs.swo.moe/torchattack](https://docs.swo.moe/torchattack/).

## Usage

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Load a pretrained model to attack from either torchvision or timm.

```python
from torchattack import AttackModel

# Load a model with `AttackModel`
model = AttackModel.from_pretrained(model_name='resnet50').to(device)
# `AttackModel` automatically attach the model's `transform` and `normalize` functions
transform, normalize = model.transform, model.normalize

# Additionally, to explicitly specify where to load the pretrained model from (timm or torchvision),
# prepend the model name with 'timm/' or 'tv/' respectively, or use the `from_timm` argument, e.g.
vit_b16 = AttackModel.from_pretrained(model_name='timm/vit_base_patch16_224').to(device)
inv_v3 = AttackModel.from_pretrained(model_name='tv/inception_v3').to(device)
pit_b = AttackModel.from_pretrained(model_name='pit_b_224', from_timm=True).to(device)
```

Initialize an attack by importing its attack class.

```python
from torchattack import FGSM, MIFGSM

# Initialize an attack
adversary = FGSM(model, normalize, device)

# Initialize an attack with extra params
adversary = MIFGSM(model, normalize, device, eps=0.03, steps=10, decay=1.0)
```

Initialize an attack by its name with `create_attack()`.

```python
from torchattack import create_attack

# Initialize FGSM attack with create_attack
adversary = create_attack('FGSM', model, normalize, device)

# Initialize PGD attack with specific eps with create_attack
adversary = create_attack('PGD', model, normalize, device, eps=0.03)

# Initialize MI-FGSM attack with extra args with create_attack
attack_args = {'steps': 10, 'decay': 1.0}
adversary = create_attack('MIFGSM', model, normalize, device, eps=0.03, **attack_args)
```

Check out [examples/](examples/mifgsm_transfer.py) and [`torchattack.evaluate.runner`](torchattack/evaluate/runner.py) for full examples.

## Attacks

We roughly categorize transferable adversarial attacks into the following categories based on their strategies to improve adversarial transferability:

- **Classic attacks**: The line of work that first proposed gradient-based adversarial attacks.
- **Gradient augmentations**: Stabilizing or augmenting the gradient flows to improve transferability.
- **Input transformations**: Applying all forms of transformations as image augmentations to inputs.
- **Feature disruption**: Disrupting intermediate features of the surrogate model.
- **Surrogate self-refinement**: Refining the surrogate model, both structure-wise and in forward/backward passes.
- **Generative modelling**: Using generative models to generate adversarial examples.
- **Others**: Other attacks that do not fit into transfer-based attacks but are important black-box attacks.
<!-- - **Ensemble surrogate refinement**: Constructing and refining surrogate ensembles. -->

We provide a detailed list of all supported attacks below.

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Class Name</th>
      <th>Publication</th>
      <th>Paper (Open Access)</th>
    </tr>
  </thead>
  <tbody>
    <!-- Classic attacks -->
    <tr>
      <th colspan="4">Classic attacks</th>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/fgsm">FGSM</a></td>
      <td><code>FGSM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2015-62B959?labelColor=2D3339" alt="ICLR 2015"></td>
      <td><a href="https://arxiv.org/abs/1412.6572">Explaining and Harnessing Adversarial Examples</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/pgd">PGD</a></td>
      <td><code>PGD</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2018-62B959?labelColor=2D3339" alt="ICLR 2018"></td>
      <td><a href="https://arxiv.org/abs/1706.06083">Towards Deep Learning Models Resistant to Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/pgdl2">PGD (L2)</a></td>
      <td><code>PGDL2</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2018-62B959?labelColor=2D3339" alt="ICLR 2018"></td>
      <td><a href="https://arxiv.org/abs/1706.06083">Towards Deep Learning Models Resistant to Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/ifgsm">I-FGSM</a></td>
      <td><code>IFGSM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2019-62B959?labelColor=2D3339" alt="ICLR 2019"></td>
      <td><a href="https://arxiv.org/abs/1607.02533">Adversarial examples in the physical world</a></td>
    </tr>
    <!-- Gradient augmentations -->
    <tr>
      <th colspan="4">Gradient augmentations</th>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/mifgsm">MI-FGSM</a></td>
      <td><code>MIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2018-1A407F?labelColor=2D3339" alt="CVPR 2018"></td>
      <td><a href="https://arxiv.org/abs/1710.06081">Boosting Adversarial Attacks with Momentum</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/nifgsm">NI-FGSM</a></td>
      <td><code>NIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2020-62B959?labelColor=2D3339" alt="ICLR 2020"></td>
      <td><a href="https://arxiv.org/abs/1908.06281">Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/vmifgsm">VMI-FGSM</a></td>
      <td><code>VMIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2021-1A407F?labelColor=2D3339" alt="CVPR 2021"></td>
      <td><a href="https://arxiv.org/abs/2103.15571">Enhancing the Transferability of Adversarial Attacks through Variance Tuning</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/vnifgsm">VNI-FGSM</a></td>
      <td><code>VNIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2021-1A407F?labelColor=2D3339" alt="CVPR 2021"></td>
      <td><a href="https://arxiv.org/abs/2103.15571">Enhancing the Transferability of Adversarial Attacks through Variance Tuning</a></td>
    </tr>
    <tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/mig">MIG</a></td>
      <td><code>MIG</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2023-5A428D?labelColor=2D3339" alt="ICCV 2023"></td>
      <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html">Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/gra">GRA</a></td>
      <td><code>GRA</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2023-5A428D?labelColor=2D3339" alt="ICCV 2023"></td>
      <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Boosting_Adversarial_Transferability_via_Gradient_Relevance_Attack_ICCV_2023_paper.html">Boosting Adversarial Transferability via Gradient Relevance Attack</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/mumodig">MuMoDIG</a></td>
      <td><code>MuMoDIG</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2025-C8172C?labelColor=2D3339" alt="AAAI 2025"></td>
      <td><a href="https://www.arxiv.org/abs/2412.18844">Improving Integrated Gradient-based Transferable Adversarial Examples by Refining the Integration Path</a></td>
    </tr>
    <!-- Input transformations -->
    <tr>
      <th colspan="4">Input transformations</th>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/difgsm">DI-FGSM</a></td>
      <td><code>DIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2019-1A407F?labelColor=2D3339" alt="CVPR 2019"></td>
      <td><a href="https://arxiv.org/abs/1803.06978">Improving Transferability of Adversarial Examples with Input Diversity</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/tifgsm">TI-FGSM</a></td>
      <td><code>TIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2019-1A407F?labelColor=2D3339" alt="CVPR 2019"></td>
      <td><a href="https://arxiv.org/abs/1904.02884">Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/sinifgsm">SI-NI-FGSM</a></td>
      <td><code>SINIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2020-62B959?labelColor=2D3339" alt="ICLR 2020"></td>
      <td><a href="https://arxiv.org/abs/1908.06281">Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/admix">Admix</a></td>
      <td><code>Admix</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2021-5A428D?labelColor=2D3339" alt="ICCV 2021"></td>
      <td><a href="https://arxiv.org/abs/2102.00436">Admix: Enhancing the Transferability of Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/ssa">SSA</a></td>
      <td><code>SSA</code></td>
      <td><img src="https://img.shields.io/badge/ECCV-2022-E16B4C?labelColor=2D3339" alt="ECCV 2022"></td>
      <td><a href="https://arxiv.org/abs/2207.05382">Frequency Domain Model Augmentation for Adversarial Attack</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/decowa">DeCoWA</a></td>
      <td><code>DeCoWA</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2024-C8172C?labelColor=2D3339" alt="AAAI 2024"></td>
      <td><a href="https://arxiv.org/abs/2402.03951">Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/bsr">BSR</a></td>
      <td><code>BSR</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2024-1A407F?labelColor=2D3339" alt="CVPR 2024"></td>
      <td><a href="https://arxiv.org/abs/2308.10299">Boosting Adversarial Transferability by Block Shuffle and Rotation</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/l2t">L2T</a></td>
      <td><code>L2T</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2024-1A407F?labelColor=2D3339" alt="CVPR 2024"></td>
      <td><a href="https://arxiv.org/abs/2405.14077">Learning to Transform Dynamically for Better Adversarial Transferability</a></td>
    </tr>
    <!-- Feature disruption -->
    <tr>
      <th colspan="4">Feature disruption</th>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/fda">FDA</a></td>
      <td><code>FDA</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2019-5A428D?labelColor=2D3339" alt="ICCV 2019"></td>
      <td><a href="https://arxiv.org/abs/1909.04385">FDA: Feature Disruptive Attack</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/dr">DR</a></td>
      <td><code>DR</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2020-1A407F?labelColor=2D3339" alt="CVPR 2020"></td>
      <td><a href="https://arxiv.org/abs/1911.11616">Enhancing Cross-Task Black-Box Transferability of Adversarial Examples With Dispersion Reduction</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/fia">FIA</a></td>
      <td><code>FIA</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2021-5A428D?labelColor=2D3339" alt="ICCV 2021"></td>
      <td><a href="https://arxiv.org/abs/2107.14185">Feature Importance-aware Transferable Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/naa">NAA</a></td>
      <td><code>NAA</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2022-1A407F?labelColor=2D3339" alt="CVPR 2022"></td>
      <td><a href="https://arxiv.org/abs/2204.00008">Improving Adversarial Transferability via Neuron Attribution-Based Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/ilpd">ILPD</a></td>
      <td><code>ILPD</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2023-654287?labelColor=2D3339" alt="NeurIPS 2023"></td>
      <td><a href="https://arxiv.org/abs/2304.13410">Improving Adversarial Transferability via Intermediate-level Perturbation Decay</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/danaa">DANAA</a></td>
      <td><code>DANAA</code></td>
      <td><img src="https://img.shields.io/badge/ADMA-2023-005C00?labelColor=2D3339" alt="ADMA 2023"></td>
      <td><a href="https://arxiv.org/abs/2310.10427">DANAA: Towards transferable attacks with double adversarial neuron attribution</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/bfa">BFA</a></td>
      <td><code>BFA</code></td>
      <td><img src="https://img.shields.io/badge/Neurocomputing-cfb225" alt="Neurocomputing' 2024"></td>
      <td><a href="https://www.sciencedirect.com/science/article/pii/S0925231224006349">Improving the transferability of adversarial examples through black-box feature attacks</a></td>
    </tr>
    <!-- Surrogate self-refinement -->
    <tr>
      <th colspan="4">Surrogate self-refinement</th>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/sgm">SGM</a></td>
      <td><code>SGM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2020-62B959?labelColor=2D3339" alt="ICLR 2020"></td>
      <td><a href="https://arxiv.org/abs/2002.05990">Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/pnapatchout">PNA-PatchOut</a></td>
      <td><code>PNAPatchOut</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2022-C8172C?labelColor=2D3339" alt="AAAI 2022"></td>
      <td><a href="https://arxiv.org/abs/2109.04176">Towards Transferable Adversarial Attacks on Vision Transformers</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/tgr">TGR</a></td>
      <td><code>TGR</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2023-1A407F?labelColor=2D3339" alt="CVPR 2023"></td>
      <td><a href="https://arxiv.org/abs/2303.15754">Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/bpa">BPA</a></td>
      <td><code>BPA</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2023-654287?labelColor=2D3339" alt="NeurIPS 2023"></td>
      <td><a href="https://arxiv.org/abs/2306.12685">Rethinking the Backward Propagation for Adversarial Transferability</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/vdc">VDC</a></td>
      <td><code>VDC</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2024-C8172C?labelColor=2D3339" alt="AAAI 2024"></td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/28541">Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/att">ATT</a></td>
      <td><code>ATT</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2024-654287?labelColor=2D3339" alt="NeurIPS 2024"></td>
      <td><a href="https://openreview.net/forum?id=sNz7tptCH6">Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning</a></td>
    </tr>
    <!-- Ensemble surrogate refinement -->
    <!-- <tr>
      <th colspan="4">Ensemble surrogate refinement</th>
    </tr> -->
    <!-- Generative modelling -->
    <tr>
      <th colspan="4">Generative modelling</th>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/cda">CDA</a></td>
      <td><code>CDA</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2019-654287?labelColor=2D3339" alt="NeurIPS 2019"></td>
      <td><a href="https://arxiv.org/abs/1905.11736">Cross-Domain Transferability of Adversarial Perturbations</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/ltp">LTP</a></td>
      <td><code>LTP</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2021-654287?labelColor=2D3339" alt="NeurIPS 2021"></td>
      <td><a href="https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html">Learning Transferable Adversarial Perturbations</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/bia">BIA</a></td>
      <td><code>BIA</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2022-62B959?labelColor=2D3339" alt="ICLR 2022"></td>
      <td><a href="https://arxiv.org/abs/2201.11528">Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/gama">GAMA</a></td>
      <td><code>GAMA</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2022-654287?labelColor=2D3339" alt="NeurIPS 2022"></td>
      <td><a href="https://arxiv.org/abs/2209.09502">GAMA: Generative Adversarial Multi-Object Scene Attacks</a></td>
    </tr>
    <!-- Others -->
    <tr>
      <th colspan="4">Others</th>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/deepfool">DeepFool</a></td>
      <td><code>DeepFool</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2016-1A407F?labelColor=2D3339" alt="CVPR 2016"></td>
      <td><a href="https://arxiv.org/abs/1511.04599">DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/geoda">GeoDA</a></td>
      <td><code>GeoDA</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2020-1A407F?labelColor=2D3339" alt="CVPR 2020"></td>
      <td><a href="https://arxiv.org/abs/2003.06468">GeoDA: A Geometric Framework for Black-box Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="https://docs.swo.moe/torchattack/attacks/ssp">SSP</a></td>
      <td><code>SSP</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2020-1A407F?labelColor=2D3339" alt="CVPR 2020"></td>
      <td><a href="https://arxiv.org/abs/2006.04924">A Self-supervised Approach for Adversarial Robustness</a></td>
    </tr>
  </tbody>
</table>

## Development

On how to install dependencies, run tests, and build documentation. See [Development - torchattack](https://docs.swo.moe/torchattack/development/).

## License

[MIT](LICENSE)

## Related

- [Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
