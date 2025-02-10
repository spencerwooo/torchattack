from torchattack.admix import Admix
from torchattack.att import ATT
from torchattack.attack_model import AttackModel
from torchattack.bia import BIA
from torchattack.bsr import BSR
from torchattack.cda import CDA
from torchattack.create_attack import create_attack
from torchattack.decowa import DeCoWA
from torchattack.deepfool import DeepFool
from torchattack.difgsm import DIFGSM
from torchattack.dr import DR
from torchattack.fgsm import FGSM
from torchattack.fia import FIA
from torchattack.gama import GAMA
from torchattack.geoda import GeoDA
from torchattack.ilpd import ILPD
from torchattack.l2t import L2T
from torchattack.ltp import LTP
from torchattack.mifgsm import MIFGSM
from torchattack.mig import MIG
from torchattack.naa import NAA
from torchattack.nifgsm import NIFGSM
from torchattack.pgd import PGD
from torchattack.pgdl2 import PGDL2
from torchattack.pna_patchout import PNAPatchOut
from torchattack.sinifgsm import SINIFGSM
from torchattack.ssa import SSA
from torchattack.ssp import SSP
from torchattack.tgr import TGR
from torchattack.tifgsm import TIFGSM
from torchattack.vdc import VDC
from torchattack.vmifgsm import VMIFGSM
from torchattack.vnifgsm import VNIFGSM

__version__ = '1.3.0'

__all__ = [
    # Helper function to create an attack by its name
    'create_attack',
    # Optional but recommended model wrapper
    'AttackModel',
    # All supported attacks
    'Admix',
    'ATT',
    'BIA',
    'BSR',
    'CDA',
    'DeCoWA',
    'DeepFool',
    'DIFGSM',
    'DR',
    'FGSM',
    'FIA',
    'GAMA',
    'GeoDA',
    'ILPD',
    'L2T',
    'LTP',
    'MIFGSM',
    'MIG',
    'NAA',
    'NIFGSM',
    'PGD',
    'PGDL2',
    'PNAPatchOut',
    'SINIFGSM',
    'SSA',
    'SSP',
    'TGR',
    'TIFGSM',
    'VDC',
    'VMIFGSM',
    'VNIFGSM',
]

GRADIENT_NON_VIT_ATTACKS = {
    'Admix': Admix,
    'BSR': BSR,
    'DeCoWA': DeCoWA,
    'DIFGSM': DIFGSM,
    'DR': DR,
    'FGSM': FGSM,
    'FIA': FIA,
    'ILPD': ILPD,
    'L2T': L2T,
    'MIFGSM': MIFGSM,
    'MIG': MIG,
    'NAA': NAA,
    'NIFGSM': NIFGSM,
    'PGD': PGD,
    'PGDL2': PGDL2,
    'SINIFGSM': SINIFGSM,
    'SSA': SSA,
    'SSP': SSP,
    'TIFGSM': TIFGSM,
    'VMIFGSM': VMIFGSM,
    'VNIFGSM': VNIFGSM,
}
GRADIENT_VIT_ATTACKS = {
    'TGR': TGR,
    'VDC': VDC,
    'PNAPatchOut': PNAPatchOut,
    'ATT': ATT,
}
GENERATIVE_ATTACKS = {
    'BIA': BIA,
    'CDA': CDA,
    'GAMA': GAMA,
    'LTP': LTP,
}
NON_EPS_ATTACKS = {
    'GeoDA': GeoDA,
    'DeepFool': DeepFool,
}
SUPPORTED_ATTACKS = (
    GRADIENT_NON_VIT_ATTACKS
    | GRADIENT_VIT_ATTACKS
    | GENERATIVE_ATTACKS
    | NON_EPS_ATTACKS
)
