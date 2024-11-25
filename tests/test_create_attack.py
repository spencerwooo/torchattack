import pytest

from torchattack import (
    DIFGSM,
    FGSM,
    FIA,
    MIFGSM,
    NIFGSM,
    PGD,
    PGDL2,
    SINIFGSM,
    SSA,
    SSP,
    TGR,
    TIFGSM,
    VDC,
    VMIFGSM,
    VNIFGSM,
    Admix,
    DeCoWA,
    DeepFool,
    GeoDA,
    PNAPatchOut,
    create_attack,
)

expected_non_vit_attacks = {
    'DIFGSM': DIFGSM,
    'FGSM': FGSM,
    'FIA': FIA,
    'MIFGSM': MIFGSM,
    'NIFGSM': NIFGSM,
    'PGD': PGD,
    'PGDL2': PGDL2,
    'SINIFGSM': SINIFGSM,
    'SSA': SSA,
    'SSP': SSP,
    'TIFGSM': TIFGSM,
    'VMIFGSM': VMIFGSM,
    'VNIFGSM': VNIFGSM,
    'Admix': Admix,
    'DeCoWA': DeCoWA,
    'DeepFool': DeepFool,
    'GeoDA': GeoDA,
}
expected_vit_attacks = {
    'TGR': TGR,
    'VDC': VDC,
    'PNAPatchOut': PNAPatchOut,
}


@pytest.mark.parametrize(('attack_name', 'expected'), expected_non_vit_attacks.items())
def test_create_non_vit_attack_same_as_imported(
    attack_name,
    expected,
    resnet50_model,
):
    created_attacker = create_attack(attack_name, resnet50_model)
    expected_attacker = expected(resnet50_model)
    assert created_attacker == expected_attacker


@pytest.mark.parametrize(('attack_name', 'expected'), expected_vit_attacks.items())
def test_create_vit_attack_same_as_imported(
    attack_name,
    expected,
    vitb16_model,
):
    created_attacker = create_attack(attack_name, vitb16_model)
    expected_attacker = expected(vitb16_model)
    assert created_attacker == expected_attacker


def test_create_attack_with_eps(device, resnet50_model):
    eps = 0.3
    attack_cfg = {}
    attacker = create_attack(
        attack_name='FGSM',
        model=resnet50_model,
        normalize=resnet50_model.normalize,
        device=device,
        eps=eps,
        attack_cfg=attack_cfg,
    )
    assert attacker.eps == eps


def test_create_attack_with_attack_cfg_eps(device, resnet50_model):
    attack_cfg = {'eps': 0.1}
    attacker = create_attack(
        attack_name='FGSM',
        model=resnet50_model,
        normalize=resnet50_model.normalize,
        device=device,
        attack_cfg=attack_cfg,
    )
    assert attacker.eps == attack_cfg['eps']


def test_create_attack_with_both_eps_and_attack_cfg(device, resnet50_model):
    eps = 0.3
    attack_cfg = {'eps': 0.1}
    # with pytest.warns(
    #     UserWarning,
    #     match="'eps' in 'attack_cfg' (0.1) will be overwritten by the 'eps' argument value (0.3), which MAY NOT be intended.",
    # ):
    attacker = create_attack(
        attack_name='FGSM',
        model=resnet50_model,
        normalize=resnet50_model.normalize,
        device=device,
        eps=eps,
        attack_cfg=attack_cfg,
    )
    assert attacker.eps == eps


def test_create_attack_with_invalid_eps(device, resnet50_model):
    eps = 0.3
    with pytest.warns(
        UserWarning, match="parameter 'eps' is invalid in DeepFool and will be ignored."
    ):
        attacker = create_attack(
            attack_name='DeepFool',
            model=resnet50_model,
            normalize=resnet50_model.normalize,
            device=device,
            eps=eps,
        )
    assert 'eps' not in attacker.__dict__


def test_create_attack_with_invalid_attack_name(device, resnet50_model):
    with pytest.raises(
        ValueError, match="Attack 'InvalidAttack' is not supported within torchattack."
    ):
        create_attack(
            attack_name='InvalidAttack',
            model=resnet50_model,
            normalize=resnet50_model.normalize,
            device=device,
        )
