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
