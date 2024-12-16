---
title: Home
hide:
  - navigation
  - footer
---

<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
</style>

<figure markdown="span">
![torchattack](./images/torchattack.png){: style="width:600px"}
</figure>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pypi python versions](https://img.shields.io/pypi/pyversions/torchattack.svg?labelColor=2D3339)](https://pypi.python.org/pypi/torchattack)
[![pypi version](https://img.shields.io/pypi/v/torchattack.svg?labelColor=2D3339)](https://pypi.python.org/pypi/torchattack)
[![github release (latest by date)](https://img.shields.io/github/v/release/spencerwooo/torchattack?labelColor=2D3339)](https://github.com/spencerwooo/torchattack/releases/latest)
[![lint](https://github.com/spencerwooo/torchattack/actions/workflows/ci.yml/badge.svg)](https://github.com/spencerwooo/torchattack/actions/workflows/ci.yml)

_A curated list of adversarial attacks in PyTorch, with a focus on transferable black-box attacks._

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

## Next Steps

<div class="grid cards" markdown>

- :material-book-open-page-variant:{ .middle } **Usage**

    ***

    Learn how to use abstractions of pre-trained victim models, attack creations, and evaluations.

    :material-arrow-right: [Usage](./usage/index.md)

- :material-sword-cross:{ .middle } **Attacks**

    ***

    Explore the comprehensive list of adversarial attacks available in torchattack.

    :material-arrow-right: [Attacks](./attacks/index.md)

- :material-tools:{ .middle } **Development**

    ***

    On how to install dependencies, run tests, and build documentation.

    :material-arrow-right: [Development](./development.md)

</div>

## License

torchattack is licensed under the [MIT License](https://github.com/spencerwooo/torchattack/blob/main/LICENSE).
