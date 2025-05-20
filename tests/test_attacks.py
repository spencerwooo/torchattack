import pytest

import torchattack
from torchattack import ATTACK_REGISTRY
from torchattack.attack_model import AttackModel


def run_attack_test(attack_cls, device, model, x, y):
    model = model.to(device)
    normalize = model.normalize
    # attacker = attack_cls(model, normalize, device=device)
    attacker = torchattack.create_attack(attack_cls, model, normalize, device=device)
    x, y = x.to(device), y.to(device)
    x_adv = attacker(x, y)
    x_outs, x_adv_outs = model(normalize(x)), model(normalize(x_adv))
    assert x_outs.argmax(dim=1) == y
    assert x_adv_outs.argmax(dim=1) != y


@pytest.mark.parametrize(
    'attack_cls',
    [
        ac
        for ac in ATTACK_REGISTRY.values()
        if (ac.is_category('COMMON') or ac.is_category('NON_EPS'))
    ],
)
def test_common_and_non_eps_attacks(attack_cls, device, resnet50_model, data):
    x, y = data(resnet50_model.transform)
    run_attack_test(attack_cls, device, resnet50_model, x, y)


@pytest.mark.parametrize(
    'attack_cls',
    [ac for ac in ATTACK_REGISTRY.values() if ac.is_category('GRADIENT_VIT')],
)
def test_gradient_vit_attacks(attack_cls, device, vitb16_model, data):
    x, y = data(vitb16_model.transform)
    run_attack_test(attack_cls, device, vitb16_model, x, y)


@pytest.mark.parametrize(
    'attack_cls',
    [ac for ac in ATTACK_REGISTRY.values() if ac.is_category('GENERATIVE')],
)
def test_generative_attacks(attack_cls, device, resnet50_model, data):
    x, y = data(resnet50_model.transform)
    run_attack_test(attack_cls, device, resnet50_model, x, y)


@pytest.mark.parametrize(
    'model_name',
    [
        'deit_base_distilled_patch16_224',
        'pit_b_224',
        'cait_s24_224',
        'visformer_small',
    ],
)
def test_tgr_attack_all_supported_models(device, model_name, data):
    model = AttackModel.from_pretrained(model_name, from_timm=True).to(device)
    x, y = data(model.transform)
    run_attack_test(torchattack.TGR, device, model, x, y)


@pytest.mark.parametrize(
    'model_name',
    [
        'deit_base_distilled_patch16_224',
        'pit_b_224',
        'visformer_small',
    ],
)
def test_vdc_attack_all_supported_models(device, model_name, data):
    model = AttackModel.from_pretrained(model_name, from_timm=True).to(device)
    x, y = data(model.transform)
    run_attack_test(torchattack.VDC, device, model, x, y)


@pytest.mark.parametrize(
    'model_name',
    [
        'pit_b_224',
        'cait_s24_224',
        'visformer_small',
    ],
)
def test_att_attack_all_supported_models(device, model_name, data):
    model = AttackModel.from_pretrained(model_name, from_timm=True).to(device)
    x, y = data(model.transform)
    run_attack_test(torchattack.ATT, device, model, x, y)
