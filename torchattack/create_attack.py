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
    attack: Any,
    model: nn.Module | AttackModel | None = None,
    normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
    eps: float | None = None,
    weights: str | None = 'DEFAULT',
    checkpoint_path: str | None = None,
    attack_cfg: dict[str, Any] | None = None,
) -> Attack:
    """Create a torchattack instance based on the provided attack name and config.

    Args:
        attack: The attack to create, either by name or class instance.
        model: The model to be attacked. Can be an instance of nn.Module or AttackModel. Defaults to None.
        normalize: The normalization function specific to the model. Defaults to None.
        device: The device on which the attack will be executed. Defaults to None.
        eps: The epsilon value for the attack. Defaults to None.
        weights: The name of the generator weight for generative attacks. Defaults to 'DEFAULT'.
        checkpoint_path: The path to the checkpoint file for generative attacks. Defaults to None.
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
        - The `weights` and `checkpoint_path` parameters are only used for generative attacks.
    """

    if attack_cfg is None:
        attack_cfg = {}

    # Check if attack_name is supported
    attack_name = attack if isinstance(attack, str) else attack.__name__
    if attack_name not in torchattack.SUPPORTED_ATTACKS:
        raise ValueError(f"Attack '{attack_name}' is not supported within torchattack.")
    if not isinstance(attack, Attack):
        attack = getattr(torchattack, attack_name)

    # Check if eps is provided and overwrite the value in attack_cfg if present
    if eps is not None:
        if 'eps' in attack_cfg:
            attack_warn(
                "The 'eps' value provided as an argument will overwrite the existing "
                "'eps' value in 'attack_cfg'. This MAY NOT be the intended behavior."
            )
        attack_cfg['eps'] = eps

    # Check if attacks that do not require eps have eps in attack_cfg
    if attack_name in torchattack.NON_EPS_ATTACKS and 'eps' in attack_cfg:
        attack_warn(f"argument 'eps' is invalid in {attack_name} and will be ignored.")
        attack_cfg.pop('eps', None)

    # Check if non-generative attacks have weights or checkpoint_path
    if attack_name not in torchattack.GENERATIVE_ATTACKS and (
        weights != 'DEFAULT' or checkpoint_path is not None
    ):
        attack_warn(
            f"argument 'weights' and 'checkpoint_path' are only used for "
            f"generative attacks, and will be ignored for '{attack_name}'."
        )
        attack_cfg.pop('weights', None)
        attack_cfg.pop('checkpoint_path', None)

    # Special handling for generative attacks
    if attack_name in torchattack.GENERATIVE_ATTACKS:
        attack_cfg['weights'] = weights
        attack_cfg['checkpoint_path'] = checkpoint_path
        return attack(device, **attack_cfg)

    return attack(model, normalize, device, **attack_cfg)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained('resnet18', device)
    attacker = create_attack('MIFGSM', model, model.normalize, device)
    print(attacker)
