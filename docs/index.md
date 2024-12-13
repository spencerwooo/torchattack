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

<figure markdown="1">
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

!!! danger "Under Construction"

    This page is currently under heavy construction.

## Highlights

- 🛡️ A curated collection of adversarial attacks implemented in PyTorch.
- 🔍 Focuses on gradient-based transferable black-box attacks.
- 📦 Easily load pretrained models from torchvision or timm using `AttackModel`.
- 🔄 Simple interface to initialize attacks with `create_attack`.
- 🔧 Extensively typed for better code quality and safety.
- 📊 Tooling for fooling rate metrics and model evaluation in `eval`.
- 🔁 Numerous attacks reimplemented for readability and efficiency (TGR, VDC, etc.).

## Next Steps

[Usage](./usage.md) | [Attacks](./attacks/index.md)

## License

torchattack is licensed under the [MIT License](https://github.com/spencerwooo/torchattack/blob/main/LICENSE).