from typing import Any, Callable

import torch
import torch.nn as nn

import torchattack
from torchattack._attack import Attack
from torchattack.attack_model import AttackModel


def create_attack(
    attack_name: str,
    model: nn.Module | AttackModel,
    normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
    eps: float | None = None,
    attack_cfg: dict[str, Any] | None = None,
) -> Attack:
    if attack_cfg is None:
        attack_cfg = {}
    if eps is not None:
        if 'eps' in attack_cfg:
            print('Warning: `eps` in `attack_cfg` will be overwritten.')
        attack_cfg['eps'] = eps
    if attack_name in ['GeoDA', 'DeepFool']:
        if 'eps' in attack_cfg:
            print(f'Warning: `eps` is invalid in `{attack_name}` and will be ignored.')
        attack_cfg.pop('eps', None)
    attacker_cls = getattr(torchattack, attack_name)
    return attacker_cls(model, normalize, device, **attack_cfg)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained('resnet18', device)
    attacker = create_attack('MIFGSM', model, model.normalize, device)
    print(attacker)
