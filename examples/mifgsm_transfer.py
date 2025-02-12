import torch
from rich.progress import track

from torchattack import MIFGSM, AttackModel
from torchattack.eval import FoolingRateMetric, NIPSLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the white-box model
model = AttackModel.from_pretrained(model_name='resnet50').to(device)
transform, normalize = model.transform, model.normalize

# Initialize the attacker MI-FGSM
attacker = MIFGSM(model, normalize, device, eps=8 / 255)

# Load the dataset
dataloader = NIPSLoader(root='datasets/nips2017', transform=transform, batch_size=16)

# Initialize the black-box victim model (vmodel)
vmodel = AttackModel.from_pretrained(model_name='vgg19').to(device)
# Track both white-box and black-box fooling rates
frm, vfrm = (FoolingRateMetric(), FoolingRateMetric())

# Attack loop
for x, y, _ in track(dataloader, description='Evaluating attack'):
    x, y = x.to(device), y.to(device)
    x_adv = attacker(x, y)

    # Track fooling rate
    x_outs = model(normalize(x))
    adv_outs = model(normalize(x_adv))
    frm.update(y, x_outs, adv_outs)

    # Track black-box fooling rate
    vx_outs = vmodel(vmodel.normalize(x))
    vadv_outs = vmodel(vmodel.normalize(x_adv))
    vfrm.update(y, vx_outs, vadv_outs)

# Show evaluation results
cln_acc, adv_acc, fr = frm.compute()
vcln_acc, vadv_acc, vfr = vfrm.compute()
print(f'White-box (ResNet-50): {cln_acc:.2%} -> {adv_acc:.2%} (FR: {fr:.2%})')
print(f'Black-box (VGG-19): {vcln_acc:.2%} -> {vadv_acc:.2%} (FR: {vfr:.2%})')
