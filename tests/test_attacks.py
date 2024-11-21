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
)


def run_attack_test(attack_cls, device, model, x, y):
    normalize = model.normalize
    attacker = attack_cls(model, normalize, device=device)
    x, y = x.to(device), y.to(device)
    x_adv = attacker(x, y)
    x_outs, x_adv_outs = model(normalize(x)), model(normalize(x_adv))
    assert x_outs.argmax(dim=1) == y
    assert x_adv_outs.argmax(dim=1) != y


@pytest.mark.parametrize(
    'attack_cls',
    [
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
        TIFGSM,
        VMIFGSM,
        VNIFGSM,
        Admix,
        DeCoWA,
    ],
)
def test_cnn_attacks(attack_cls, device, prepare_cnn_and_data):
    model, (x, y) = prepare_cnn_and_data
    run_attack_test(attack_cls, device, model, x, y)


@pytest.mark.parametrize(
    'attack_cls',
    [
        TGR,
        VDC,
        PNAPatchOut,
    ],
)
def test_vit_attacks(attack_cls, device, prepare_vit_and_data):
    model, (x, y) = prepare_vit_and_data
    run_attack_test(attack_cls, device, model, x, y)


@pytest.mark.parametrize('attack_cls', [DeepFool, GeoDA])
def test_non_gradient_attacks(attack_cls, device, prepare_cnn_and_data):
    model, (x, y) = prepare_cnn_and_data
    run_attack_test(attack_cls, device, model, x, y)
