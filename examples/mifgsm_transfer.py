import torch
from rich.progress import track

from torchattack import MIFGSM, AttackModel
from torchattack.eval import FoolingRateMetric, NIPSLoader, save_image_batch

bs = 16
eps = 8 / 255
root = 'datasets/nips2017'
save_dir = 'outputs'
model_name = 'resnet50'
victim_names = ['vgg11', 'inception_v3']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the white-box model, fooling rate metric, and dataloader
model = AttackModel.from_pretrained(model_name).to(device)
frm = FoolingRateMetric()
dataloader = NIPSLoader(root, transform=model.transform, batch_size=bs)

# Initialize the attacker MI-FGSM
attacker = MIFGSM(model, model.normalize, device, eps)

# Attack loop and save generated adversarial examples
for x, y, fnames in track(dataloader, description=f'Evaluating white-box {model_name}'):
    x, y = x.to(device), y.to(device)
    x_adv = attacker(x, y)

    # Track fooling rate
    x_outs = model(model.normalize(x))
    adv_outs = model(model.normalize(x_adv))
    frm.update(y, x_outs, adv_outs)

    # Save adversarial examples to `save_dir`
    save_image_batch(x_adv, save_dir, fnames)

# Evaluate fooling rate
cln_acc, adv_acc, fr = frm.compute()
print(f'White-box ({model_name}): {cln_acc:.2%} -> {adv_acc:.2%} (FR: {fr:.2%})')

# For all victim models
for vname in victim_names:
    # Initialize the black-box model, fooling rate metric, and dataloader
    vmodel = AttackModel.from_pretrained(model_name=vname).to(device)
    vfrm = FoolingRateMetric()

    # Create relative transform (relative to the white-box model) to avoid degrading the
    # effectiveness of adversarial examples through image transformations
    vtransform = vmodel.create_relative_transform(model)

    # Load the clean and adversarial examples from separate dataloaders
    clnloader = NIPSLoader(root=root, transform=vmodel.transform, batch_size=bs)
    advloader = NIPSLoader(
        image_root=save_dir,
        pairs_path=f'{root}/images.csv',
        transform=vtransform,
        batch_size=bs,
    )

    # Black-box evaluation loop
    for (x, y, _), (xadv, _, _) in track(
        zip(clnloader, advloader),
        total=len(clnloader),
        description=f'Evaluating black-box {vname}',
    ):
        x, y, xadv = x.to(device), y.to(device), xadv.to(device)
        vx_outs = vmodel(vmodel.normalize(x))
        vadv_outs = vmodel(vmodel.normalize(xadv))
        vfrm.update(y, vx_outs, vadv_outs)

    # Evaluate fooling rate
    vcln_acc, vadv_acc, vfr = vfrm.compute()
    print(f'Black-box ({vname}): {vcln_acc:.2%} -> {vadv_acc:.2%} (FR: {vfr:.2%})')
