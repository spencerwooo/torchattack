def run_attack(attack, kwargs: dict) -> None:
    """Helper function to run attacks in `__main__`.

    Example:

        >>> from torchattack import FGSM
        >>> cfg = {"eps": 8 / 255, "clip_min": 0.0, "clip_max": 1.0}
        >>> run_attack(attack=FGSM, kwargs=cfg)

    Args:
        attack: The attack class to initialize.
        kwargs: A dict of keyword arguments passed to the attack class.
    """

    from contextlib import suppress

    import torch
    from torchvision.io import write_png
    from torchvision.models import ResNet50_Weights, resnet50
    from torchvision.utils import make_grid

    from torchattack.dataset import T_NORMALIZE, T_RESIZE_224, NIPSLoader

    with suppress(ImportError):
        from rich import print
        from rich.progress import track

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)
    dataloader = NIPSLoader(path="data/nips2017", batch_size=16, transform=T_RESIZE_224)

    transform = T_NORMALIZE

    total, acc_clean, acc_adv = len(dataloader.dataset), 0, 0  # type: ignore
    attacker = attack(model=model, transform=transform, device=device, **kwargs)
    print(attacker)

    # Wrap dataloader with rich.progress.track if available
    try:
        dataloader = track(dataloader, description="Attacking")  # type: ignore
    except NameError:
        print("Running attack ... (install `rich` for progress bar)")

    for i, (_, images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        adv_images = attacker(images, labels)

        clean_outs = model(transform(images)).argmax(dim=1)
        adv_outs = model(transform(adv_images)).argmax(dim=1)

        acc_clean += (clean_outs == labels).sum().item()
        acc_adv += (adv_outs == labels).sum().item()

        if i == 12:
            saved_imgs = adv_images.detach().cpu().mul(255).to(torch.uint8)
            img_grid = make_grid(saved_imgs, nrow=8)
            write_png(img_grid, "adv_batch_12.png")

    print(f"Accuracy (clean vs adversarial): {acc_clean / total} vs {acc_adv / total}")
