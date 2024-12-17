from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

import torchattack
from torchattack._attack import Attack
from torchattack.attack_model import AttackModel


def create_attack(
    attack: Union['Attack', str],
    model: Optional[Union[nn.Module, AttackModel]] = None,
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    *,
    eps: Optional[float] = None,
    **kwargs: Any,
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
    """

    # Determine attack name and check if it is supported
    attack_name = attack if isinstance(attack, str) else attack.__name__  # type: ignore[attr-defined]
    if attack_name not in torchattack.SUPPORTED_ATTACKS:
        raise ValueError(f"Attack '{attack_name}' is not supported within torchattack.")
    # Get attack class if passed as a string
    attack_cls = (
        getattr(torchattack, attack_name) if isinstance(attack, str) else attack
    )

    # `eps` is explicitly set as it is such a common argument
    # All other arguments should be passed as keyword arguments
    if eps is not None:
        kwargs['eps'] = eps

    # Special handling for generative attacks
    attacker: Attack = (
        attack_cls(device=device, **kwargs)
        if attack_name in torchattack.GENERATIVE_ATTACKS
        else attack_cls(model=model, normalize=normalize, device=device, **kwargs)
    )
    return attacker


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained('resnet18', device)
    normalize = model.normalize
    print(create_attack('MIFGSM', model, normalize, device, eps=0.1, steps=40))
    print(create_attack('CDA', device=device, weights='VGG19_IMAGENET'))
    print(create_attack('DeepFool', model, normalize, device, num_classes=20))
