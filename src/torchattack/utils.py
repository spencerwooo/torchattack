def run_attack(attack, attack_cfg, model='resnet50', samples=100, batch_size=8) -> None:
    """Helper function to run attacks in `__main__`.

    Example:

        >>> from torchattack import FGSM
        >>> cfg = {"eps": 8 / 255, "clip_min": 0.0, "clip_max": 1.0}
        >>> run_attack(attack=FGSM, attack_cfg=cfg)

    Args:
        attack: The attack class to initialize.
        attack_cfg: A dict of keyword arguments passed to the attack class.
        model: The model to attack. Defaults to "resnet50".
        samples: Max number of samples to attack. Defaults to 100.
    """

    from contextlib import suppress

    import torch
    import torchvision as tv

    from torchattack.dataset import NIPSLoader

    # Try to import rich for progress bar
    with suppress(ImportError):
        from rich import print
        from rich.progress import track

    # Set up model and dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = tv.models.get_model(name=model, weights='DEFAULT').to(device).eval()
    dataloader = NIPSLoader(
        path='data/nips2017',
        batch_size=batch_size,
        transform=tv.transforms.Resize(size=224, antialias=True),
        max_samples=samples,
    )

    # Set up attack and trackers
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    total, acc_clean, acc_adv = len(dataloader.dataset), 0, 0  # type: ignore
    attacker = attack(model=model, transform=normalize, device=device, **attack_cfg)
    print(attacker)

    # Wrap dataloader with rich.progress.track if available
    try:
        dataloader = track(dataloader, description='Attacking')  # type: ignore
    except NameError:
        print('Running attack ... (install `rich` for progress bar)')

    # Run attack over the dataset (100 images by default)
    for i, (images, labels, _) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Adversarial images are created here
        adv_images = attacker(images, labels)

        # Track accuracy
        clean_outs = model(normalize(images)).argmax(dim=1)
        adv_outs = model(normalize(adv_images)).argmax(dim=1)

        acc_clean += (clean_outs == labels).sum().item()
        acc_adv += (adv_outs == labels).sum().item()

        # Save the 4th batch of adversarial images
        if i == 4:
            saved_imgs = adv_images.detach().cpu().mul(255).to(torch.uint8)
            img_grid = tv.utils.make_grid(saved_imgs, nrow=4)
            tv.io.write_png(img_grid, 'adv_batch_4.png')

    print(f'Accuracy (clean vs adversarial): {acc_clean / total} vs {acc_adv / total}')
