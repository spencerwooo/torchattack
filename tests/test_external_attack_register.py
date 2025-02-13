from torchattack import ATTACK_REGISTRY, Attack, create_attack, register_attack


@register_attack()
class ExternalNewAttack(Attack):
    def __init__(self, model, normalize, device):
        super().__init__(model, normalize, device)

    def forward(self, x):
        return x


def test_external_attack_registered():
    assert 'ExternalNewAttack' in ATTACK_REGISTRY
    assert issubclass(ATTACK_REGISTRY['ExternalNewAttack'], Attack)
    assert ATTACK_REGISTRY['ExternalNewAttack'].attack_name == 'ExternalNewAttack'
    assert ATTACK_REGISTRY['ExternalNewAttack'].is_category('COMMON')


def test_external_attack_can_be_created():
    ea = create_attack('ExternalNewAttack', model=None, normalize=None, device=None)
    assert isinstance(ea, ExternalNewAttack)
