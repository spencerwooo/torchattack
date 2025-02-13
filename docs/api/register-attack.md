---
status: new
---

# Register attack

!!! tip "New in 1.4.0"
    The `register_attack()` decorator was introduced in [v1.4.0](https://github.com/spencerwooo/torchattack/releases/tag/v1.4.0).

`register_attack()` is a decorator that registers an attack to the attack registry. This allows external attacks to be recognized by `create_attack()`.

The attack registry resides at `ATTACK_REGISTRY`. This registry is populated at import time. To register an additional attack, simply decorate the attack class with `@register_attack()`.

```python
from torchattack import Attack, register_attack


@register_attack()
class MyNewAttack(Attack):
    def __init__(self, model, normalize, device):
        super().__init__(model, normalize, device)

    def forward(self, x):
        return x
```

Afterwards, the attack can be accessed in the same manner as the built-in attacks.

```python
import torch
from torchattack import create_attack, AttackModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttackModel.from_pretrained('resnet50').to(device)
attacker = create_attack('MyNewAttack', model=model, normalize=model.normalize, device=device)
```

::: torchattack.register_attack
