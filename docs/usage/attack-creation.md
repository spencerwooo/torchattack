# Creating and running an attack

After [setting up a pretrained model](./attack-model.md), we can now create an attack.

## Importing the attack class

Attacks in torchattack are exposed as classes.

To create the classic attack — Fast Gradient Sign Method ([FGSM](../attacks/fgsm.md)), for instance, we can import the `FGSM` class from torchattack.

```python hl_lines="2 8"
import torch
from torchattack import AttackModel, FGSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttackModel.from_pretrained(model_name='resnet50', device=device)
transform, normalize = model.transform, model.normalize

attack = FGSM(model, normalize, device)
```

!!! info "Note the three arguments, `model`, `normalize`, and `device`."

    - `model` specifies the pretrained model to attack, often acting as the surrogate model for transferable black-box attacks.
    - `normalize` is the normalize function associated with the model's weight, that we automatically resolved earlier.
    - `device` is the device to run the attack on.

    All attacks in torchattack, if not explicitly stated, require these three arguments. (1)
    { .annotate }

    1. This is the [*thin layer of abstraction*](./index.md) that we were talking about.

Most adversarial attacks, especially gradient-based attacks, are projected onto the $\ell_p$ ball, where $p$ is the norm, to restrict the magnitude of the perturbation and maintain imperceptibility.

Attacks like FGSM takes an additional argument, `eps`, to specify the radius $\varepsilon$ of the $\ell_\infty$ ball.

For an 8-bit image with pixel values in the range of $[0, 255]$, the tensor represented image is within $[0, 1]$ (after division by 255). **As such, common values for `eps` are `8/255` or `16/255`, or simply `0.03`.**

```python
attack = FGSM(model, normalize, device, eps=8 / 255)
```

Different attacks hold different arguments. For instance, the Momentum-Iterative FGSM ([MI-FGSM](../attacks/mifgsm.md)) attack accepts the additional `steps` and `decay` as arguments.

```python
from torchattack import MIFGSM

attack = MIFGSM(model, normalize, device, eps=8 / 255, steps=10, decay=1.0)
```

Finally, please take a look at the actual implementation of FGSM here :octicons-arrow-right-24: [`torchattack.FGSM`][torchattack.FGSM] (expand collapsed `fgsm.py` source code).

## The `create_attack()` method

torchattack provides an additional helper to create attacks by its name.

```python
from torchattack import create_attack
```

To initialize the same FGSM attack.

```python
attack = create_attack('FGSM', model, normalize, device)
```

To specify the common `eps` argument.

```python
attack = create_attack('FGSM', model, normalize, device, eps=8 / 255)
```

For additional attack specific arguments, pass them as keyword arguments.

```python
attack = create_attack('MIFGSM', model, normalize, device, eps=8 / 255, steps=10, decay=1.0)
```

## Running the attack

## API Reference

::: torchattack.create_attack.create_attack
    options:
        heading_level: 3
