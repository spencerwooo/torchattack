def run_attack(attack, kwargs: dict) -> None:
    """Helper function to run attacks in `__main__`.

    Args:
        attack: The attack class.
        kwargs: Keyword arguments. Defaults to {}.
    """

    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    from torchattack.dataset import IMAGENET_TRANSFORM, NIPSLoader

    try:
        from rich import print
        from rich.progress import track
    except ImportError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)
    dataloader = NIPSLoader(path="data/nips2017", batch_size=16)

    transform = IMAGENET_TRANSFORM

    total, acc_clean, acc_adv = len(dataloader.dataset), 0, 0  # type: ignore
    attacker = attack(model=model, transform=transform, device=device, **kwargs)
    print(attacker)

    # Wrap dataloader with rich.progress.track if available
    try:
        dataloader = track(dataloader, description="Attacking")  # type: ignore
    except NameError:
        print("Running attack ... (install `rich` for progress bar)")

    for _, images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        adv_images = attacker(images, labels)

        clean_outs = model(transform(images)).argmax(dim=1)
        adv_outs = model(transform(adv_images)).argmax(dim=1)

        acc_clean += (clean_outs == labels).sum().item()
        acc_adv += (adv_outs == labels).sum().item()

    print(f"Accuracy (clean vs adversarial): {acc_clean / total} vs {acc_adv / total}")
