from typing import Any, Callable, Union

import torch
import torch.nn as nn

import torchattack
from torchattack._attack import Attack
from torchattack.attack_model import AttackModel


def attack_warn(message: str) -> None:
    from warnings import warn

    warn(message, category=UserWarning, stacklevel=2)


def attack_arg_override_warn(arg_name: str) -> None:
    attack_warn(
        f"The '{arg_name}' value provided as an argument will overwrite the existing "
        f"'{arg_name}' value in 'attack_args'. This MAY NOT be the intended behavior."
    )


def create_attack(
    attack: Union['Attack', str],
    model: Union[nn.Module, AttackModel, None] = None,
    normalize: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
    device: Union[torch.device, None] = None,
    eps: Union[float, None] = None,
    weights: Union[str, None] = 'DEFAULT',
    checkpoint_path: Union[str, None] = None,
    attack_args: Union[dict[str, Any], None] = None,
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
        attack_args: Additional config parameters for the attack. Defaults to None.

    Returns:
        Attack: An instance of the specified attack.

    Raises:
        ValueError: If the specified attack name is not supported within torchattack.

    Notes:
        - If `eps` is provided and also present in `attack_args`, a warning will be
          issued and the value in `attack_args` will be overwritten.
        - For certain attacks like 'GeoDA' and 'DeepFool', the `eps` parameter is
          invalid and will be ignored if present in `attack_args`.
        - The `weights` and `checkpoint_path` parameters are only used for generative attacks.
    """

    if attack_args is None:
        attack_args = {}

    # Determine attack name and check if it is supported
    attack_name = attack if isinstance(attack, str) else attack.__name__  # type: ignore[attr-defined]
    if attack_name not in torchattack.SUPPORTED_ATTACKS:
        raise ValueError(f"Attack '{attack_name}' is not supported within torchattack.")
    attack_cls: 'Attack' = (
        getattr(torchattack, attack_name) if isinstance(attack, str) else attack
    )

    # Check if eps is provided and overwrite the value in attack_args if present
    if eps is not None:
        if 'eps' in attack_args:
            attack_arg_override_warn('eps')
        attack_args['eps'] = eps

    # Check if attacks that do not require eps have eps in attack_args
    if attack_name in torchattack.NON_EPS_ATTACKS and 'eps' in attack_args:
        attack_warn(f"argument 'eps' is invalid in {attack_name} and will be ignored.")
        attack_args.pop('eps', None)

    # Check if non-generative attacks have weights or checkpoint_path
    if attack_name not in torchattack.GENERATIVE_ATTACKS and (
        weights != 'DEFAULT' or checkpoint_path is not None
    ):
        attack_warn(
            f"argument 'weights' and 'checkpoint_path' are only used for "
            f"generative attacks, and will be ignored for '{attack_name}'."
        )
        attack_args.pop('weights', None)
        attack_args.pop('checkpoint_path', None)

    # Special handling for generative attacks
    if attack_name in torchattack.GENERATIVE_ATTACKS:
        if weights != 'DEFAULT':
            if 'weights' in attack_args:
                attack_arg_override_warn('weights')
            attack_args['weights'] = weights
        if checkpoint_path is not None:
            if 'checkpoint_path' in attack_args:
                attack_arg_override_warn('checkpoint_path')
            attack_args['checkpoint_path'] = checkpoint_path
        return attack_cls(device, **attack_args)  # type: ignore[no-any-return]

    return attack_cls(model, normalize, device, **attack_args)  # type: ignore[no-any-return]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained('resnet18', device)
    attacker = create_attack('MIFGSM', model, model.normalize, device)
    print(attacker)
