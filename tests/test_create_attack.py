import pytest

import torchattack
from torchattack import create_attack


@pytest.mark.parametrize(
    ('attack_name', 'expected'),
    [i for i in torchattack.GRADIENT_NON_VIT_ATTACKS.items() if i[0] != 'DR'],
)
def test_create_non_vit_attack_same_as_imported(
    attack_name,
    expected,
    resnet50_model,
):
    created_attacker = create_attack(attack_name, resnet50_model)
    expected_attacker = expected(resnet50_model)
    assert created_attacker == expected_attacker


def test_create_dr_attack_same_as_imported(
    vgg16_model,
    data,
):
    created_attacker = create_attack('DR', vgg16_model)
    expected_attacker = torchattack.DR(vgg16_model)
    assert created_attacker == expected_attacker


@pytest.mark.parametrize(
    ('attack_name', 'expected'), torchattack.GRADIENT_VIT_ATTACKS.items()
)
def test_create_vit_attack_same_as_imported(
    attack_name,
    expected,
    vitb16_model,
):
    created_attacker = create_attack(attack_name, vitb16_model)
    expected_attacker = expected(vitb16_model)
    assert created_attacker == expected_attacker


@pytest.mark.parametrize(
    ('attack_name', 'expected'), torchattack.GENERATIVE_ATTACKS.items()
)
def test_create_generative_attack_same_as_imported(attack_name, expected):
    created_attacker = create_attack(attack_name)
    expected_attacker = expected()
    assert created_attacker == expected_attacker


def test_create_attack_with_eps(device, resnet50_model):
    eps = 0.3
    attack_args = {}
    attacker = create_attack(
        attack='FGSM',
        model=resnet50_model,
        normalize=resnet50_model.normalize,
        device=device,
        eps=eps,
        attack_args=attack_args,
    )
    assert attacker.eps == eps


def test_create_attack_with_attack_args_eps(device, resnet50_model):
    attack_args = {'eps': 0.1}
    attacker = create_attack(
        attack='FGSM',
        model=resnet50_model,
        normalize=resnet50_model.normalize,
        device=device,
        attack_args=attack_args,
    )
    assert attacker.eps == attack_args['eps']


def test_create_attack_with_both_eps_and_attack_args(device, resnet50_model):
    eps = 0.3
    attack_args = {'eps': 0.1}
    with pytest.warns(
        UserWarning,
        match="The 'eps' value provided as an argument will overwrite the existing "
        "'eps' value in 'attack_args'. This MAY NOT be the intended behavior.",
    ):
        attacker = create_attack(
            attack='FGSM',
            model=resnet50_model,
            normalize=resnet50_model.normalize,
            device=device,
            eps=eps,
            attack_args=attack_args,
        )
    assert attacker.eps == eps


def test_create_attack_with_both_weights_and_attack_args(device):
    weights = 'VGG19_IMAGENET'
    attack_args = {'weights': 'VGG19_IMAGENET'}
    with pytest.warns(
        UserWarning,
        match="The 'weights' value provided as an argument will "
        "overwrite the existing 'weights' value in 'attack_args'. "
        'This MAY NOT be the intended behavior.',
    ):
        attacker = create_attack(
            attack='CDA',
            device=device,
            weights=weights,
            attack_args=attack_args,
        )
    assert attacker.weights == weights


def test_create_attack_with_invalid_eps(device, resnet50_model):
    eps = 0.3
    with pytest.warns(
        UserWarning, match="argument 'eps' is invalid in DeepFool and will be ignored."
    ):
        attacker = create_attack(
            attack='DeepFool',
            model=resnet50_model,
            normalize=resnet50_model.normalize,
            device=device,
            eps=eps,
        )
    assert 'eps' not in attacker.__dict__


def test_create_attack_with_weights_and_checkpoint_path(device):
    weights = 'VGG19_IMAGENET'
    checkpoint_path = 'path/to/checkpoint'
    attack_args = {}
    with pytest.warns(
        UserWarning,
        match="argument 'weights' and 'checkpoint_path' are only used for "
        "generative attacks, and will be ignored for 'FGSM'.",
    ):
        attacker = create_attack(
            attack='FGSM',
            device=device,
            weights=weights,
            checkpoint_path=checkpoint_path,
            attack_args=attack_args,
        )
    assert 'weights' not in attacker.__dict__
    assert 'checkpoint_path' not in attacker.__dict__


def test_create_attack_with_invalid_attack_name(device, resnet50_model):
    with pytest.raises(
        ValueError, match="Attack 'InvalidAttack' is not supported within torchattack."
    ):
        create_attack(
            attack='InvalidAttack',
            model=resnet50_model,
            normalize=resnet50_model.normalize,
            device=device,
        )
