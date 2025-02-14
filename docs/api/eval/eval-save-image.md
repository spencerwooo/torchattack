---
status: new
---

# Saving images and adversarial examples

!!! tip "New in 1.5.0"
    `save_image_batch` was first introduced in [v1.5.0](https://github.com/spencerwooo/torchattack/releases/tag/v1.5.0).

To avoid degrading the effectiveness of adversarial perturbation through image saving to and opening from disk, use `save_image_batch`.

As a rule of thumb, we recommend saving images as PNGs, as they better keep the image quality than JPEGs. To compare, in the unit tests of torchattack, we use:

- a tolerance of `4e-3` for PNGs, which approx. to $\varepsilon = 1 / 255$ in the $\ell_\infty$ norm, and
- a tolerance of `8e-3` for JPEGs, which approx. to $\varepsilon = 2 / 255$.

A commonly used perturbation magnitude is $\varepsilon = 8 / 255$, for reference.

::: torchattack.eval.save_image_batch
