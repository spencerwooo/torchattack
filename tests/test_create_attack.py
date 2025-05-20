import pytest

from torchattack import ATTACK_REGISTRY, create_attack


@pytest.mark.parametrize(
    ('attack_name', 'expected'),
    [(an, ac) for an, ac in ATTACK_REGISTRY.items() if ac.is_category('COMMON')],
)
def test_create_non_vit_attack_same_as_imported(
    attack_name,
    expected,
    resnet50_model,
):
    created_attacker = create_attack(attack_name, resnet50_model)
    expected_attacker = expected(resnet50_model)
    assert created_attacker == expected_attacker


@pytest.mark.parametrize(
    ('attack_name', 'expected'),
    [(an, ac) for an, ac in ATTACK_REGISTRY.items() if ac.is_category('GRADIENT_VIT')],
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
    ('attack_name', 'expected'),
    [(an, ac) for an, ac in ATTACK_REGISTRY.items() if ac.is_category('GENERATIVE')],
)
def test_create_generative_attack_same_as_imported(attack_name, expected):
    created_attacker = create_attack(attack_name)
    expected_attacker = expected()
    assert created_attacker == expected_attacker


def test_create_attack_with_eps(device, resnet50_model):
    eps = 0.3
    attacker = create_attack(
        attack='FGSM',
        model=resnet50_model,
        normalize=resnet50_model.normalize,
        device=device,
        eps=eps,
    )
    assert attacker.eps == eps


def test_create_attack_with_extra_args(device, resnet50_model):
    attack_args = {'eps': 0.1, 'steps': 40, 'alpha': 0.01, 'decay': 0.9}
    attacker = create_attack(
        attack='MIFGSM',
        model=resnet50_model,
        normalize=resnet50_model.normalize,
        device=device,
        **attack_args,
    )
    assert attacker.eps == attack_args['eps']
    assert attacker.steps == attack_args['steps']
    assert attacker.alpha == attack_args['alpha']
    assert attacker.decay == attack_args['decay']


def test_create_attack_with_invalid_eps(device, resnet50_model):
    with pytest.raises(TypeError):
        create_attack(
            attack='DeepFool',
            model=resnet50_model,
            normalize=resnet50_model.normalize,
            device=device,
            eps=0.03,
        )


def test_create_attack_cda_with_weights(device):
    weights = 'VGG19_IMAGENET'
    attacker = create_attack(
        attack='CDA',
        device=device,
        weights=weights,
    )
    assert attacker.weights == weights


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
