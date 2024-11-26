from typing import Any, Callable

import torch
import torch.nn as nn

import torchattack
from torchattack._attack import Attack
from torchattack.attack_model import AttackModel


def attack_warn(message: str) -> None:
    from warnings import warn

    warn(message, category=UserWarning, stacklevel=2)


def create_attack(
    attack_name: str,
    model: nn.Module | AttackModel | None = None,
    normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
    eps: float | None = None,
    attack_cfg: dict[str, Any] | None = None,
) -> Attack:
    """Create a torchattack instance based on the provided attack name and config.

    Args:
        attack_name: The name of the attack to create.
        model: The model to be attacked.
        normalize: The normalization function specific to the model. Defaults to None.
        device: The device on which the attack will be executed. Defaults to None.
        eps: The epsilon value for the attack. Defaults to None.
        attack_cfg: Additional config parameters for the attack. Defaults to None.

    Returns:
        Attack: An instance of the specified attack.

    Raises:
        ValueError: If the specified attack name is not supported within torchattack.

    Notes:
        - If `eps` is provided and also present in `attack_cfg`, a warning will be
          issued and the value in `attack_cfg` will be overwritten.
        - For certain attacks like 'GeoDA' and 'DeepFool', the `eps` parameter is
          invalid and will be ignored if present in `attack_cfg`.
    """

    if attack_cfg is None:
        attack_cfg = {}
    if eps is not None:
        if 'eps' in attack_cfg:
            attack_warn(
                f"'eps' in 'attack_cfg' ({attack_cfg['eps']}) will be overwritten "
                f"by the 'eps' argument value ({eps}), which MAY NOT be intended."
            )
        attack_cfg['eps'] = eps
    if attack_name in ['GeoDA', 'DeepFool'] and 'eps' in attack_cfg:
        attack_warn(f"parameter 'eps' is invalid in {attack_name} and will be ignored.")
        attack_cfg.pop('eps', None)
    if not hasattr(torchattack, attack_name):
        raise ValueError(f"Attack '{attack_name}' is not supported within torchattack.")
    attacker_cls: Attack = getattr(torchattack, attack_name)
    if attack_name in ['BIA', 'CDA']:
        return attacker_cls(device, **attack_cfg)
    return attacker_cls(model, normalize, device, **attack_cfg)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained('resnet18', device)
    attacker = create_attack('MIFGSM', model, model.normalize, device)
    print(attacker)
