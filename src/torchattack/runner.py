from contextlib import suppress
from typing import Any

import torch
import torchvision as tv

from torchattack.dataset import NIPSLoader


class FoolingRateMetric:
    """Fooling rate metric tracker."""

    def __init__(self) -> None:
        self.total_count = torch.tensor(0)
        self.clean_count = torch.tensor(0)
        self.adv_count = torch.tensor(0)

    def update(
        self, labels: torch.Tensor, clean_logits: torch.Tensor, adv_logits: torch.Tensor
    ) -> None:
        """Update metric tracker during attack progress.

        Args:
            labels (torch.Tensor): Ground truth labels.
            clean_logits (torch.Tensor): Prediction logits for clean samples.
            adv_logits (torch.Tensor): Prediction logits for adversarial samples.
        """

        self.total_count += labels.numel()
        self.clean_count += (clean_logits.argmax(dim=1) == labels).sum().item()
        self.adv_count += (adv_logits.argmax(dim=1) == labels).sum().item()

    def compute_fooling_rate(self) -> torch.Tensor:
        return (self.clean_count - self.adv_count) / self.clean_count

    def compute_adv_accuracy(self) -> torch.Tensor:
        return self.adv_count / self.total_count

    def compute_clean_accuracy(self) -> torch.Tensor:
        return self.clean_count / self.total_count


def run_attack(
    attack: Any,
    attack_cfg: dict | None = None,
    model_name: str = 'resnet50',
    max_samples: int = 100,
    batch_size: int = 16,
) -> None:
    """Helper function to run attacks in `__main__`.

    Example:

        >>> from torchattack import FGSM
        >>> cfg = {"eps": 8 / 255, "clip_min": 0.0, "clip_max": 1.0}
        >>> run_attack(attack=FGSM, attack_cfg=cfg)

    Args:
        attack: The attack class to initialize.
        attack_cfg: A dict of keyword arguments passed to the attack class.
        model_name: The torchvision model to attack. Defaults to "resnet50".
        max_samples: Max number of samples to attack. Defaults to 100.
    """

    # Try to import rich for progress bar
    with suppress(ImportError):
        from rich import print
        from rich.progress import track

    if attack_cfg is None:
        attack_cfg = {}

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = tv.models.get_model(name=model_name, weights='DEFAULT').to(device).eval()

    # Setup transforms and normalization
    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize([256]),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
        ]
    )
    normalize = tv.transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

    # Set up dataloader
    dataloader = NIPSLoader(
        root='datasets/nips2017',
        batch_size=batch_size,
        transform=transform,
        max_samples=max_samples,
    )

    # Set up attack and trackers
    frm = FoolingRateMetric()
    attacker = attack(model=model, normalize=normalize, device=device, **attack_cfg)
    print(attacker)

    # Wrap dataloader with rich.progress.track if available
    try:
        dataloader = track(dataloader, description='Attacking')  # type: ignore
    except NameError:
        print('Running attack ... (install `rich` for progress bar)')

    # Run attack over the dataset (100 images by default)
    for i, (x, y, _) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Adversarial images are created here
        advs = attacker(x, y)

        # Track accuracy
        cln_outs = model(normalize(x))
        adv_outs = model(normalize(advs))
        frm.update(y, cln_outs, adv_outs)

        # Save first batch of adversarial images
        if i == 0:
            saved_imgs = advs.detach().cpu().mul(255).to(torch.uint8)
            img_grid = tv.utils.make_grid(saved_imgs, nrow=4)
            tv.io.write_png(img_grid, 'adv_batch_0.png')

    # Print results
    print(f'Clean accuracy: {frm.compute_clean_accuracy():.2%}')
    print(f'Adversarial accuracy: {frm.compute_adv_accuracy():.2%}')
    print(f'Fooling rate: {frm.compute_fooling_rate():.2%}')
