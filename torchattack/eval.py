from typing import Any

import torch

from torchattack.attack_model import AttackModel


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
            labels: Ground truth labels.
            clean_logits: Prediction logits for clean samples.
            adv_logits: Prediction logits for adversarial samples.
        """

        self.total_count += labels.numel()
        self.clean_count += (clean_logits.argmax(dim=1) == labels).sum().item()
        self.adv_count += (adv_logits.argmax(dim=1) == labels).sum().item()

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the fooling rate and related metrics.

        Returns:
            A tuple of torch.Tensors containing the clean accuracy, adversarial
            accuracy, and fooling rate computed, respectively.
        """
        return (
            self.clean_count / self.total_count,
            self.adv_count / self.total_count,
            (self.clean_count - self.adv_count) / self.clean_count,
        )


def run_attack(
    attack: Any,
    attack_cfg: dict | None = None,
    model_name: str = 'resnet50',
    victim_model_names: list[str] | None = None,
    max_samples: int = 100,
    batch_size: int = 16,
    from_timm: bool = False,
) -> None:
    """Helper function to run attacks in `__main__`.

    Example:
        >>> from torchattack import FGSM
        >>> cfg = {"eps": 8 / 255, "clip_min": 0.0, "clip_max": 1.0}
        >>> run_attack(attack=FGSM, attack_cfg=cfg)

    Args:
        attack: The attack class to initialize.
        attack_cfg: A dict of keyword arguments passed to the attack class.
        model_name: The surrogate model to attack. Defaults to "resnet50".
        victim_model_names: A list of the victim black-box models to attack. Defaults to None.
        max_samples: Max number of samples to attack. Defaults to 100.
        batch_size: Batch size for the dataloader. Defaults to 16.
        from_timm: Use timm to load the model. Defaults to True.
    """

    import torchvision as tv
    from rich import print
    from rich.progress import track

    from torchattack.dataset import NIPSLoader

    if attack_cfg is None:
        attack_cfg = {}

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained(model_name, device, from_timm)
    transform, normalize = model.transform, model.normalize

    # Set up dataloader
    dataloader = NIPSLoader(
        root='datasets/nips2017',
        batch_size=batch_size,
        transform=transform,
        max_samples=max_samples,
    )
    dataloader = track(dataloader, description='Attacking')  # type: ignore

    # Set up attack and trackers
    frm = FoolingRateMetric()
    attacker = attack(
        # Pass the original PyTorch model instead of the wrapped one if the attack
        # requires access to the model's intermediate layers or other attributes that
        # are not exposed by the AttackModel wrapper.
        model=model,
        normalize=normalize,
        device=device,
        **attack_cfg,
    )
    print(attacker)

    # Setup victim models if provided
    if victim_model_names:
        victim_models = [
            AttackModel.from_pretrained(name, device, from_timm)
            for name in victim_model_names
        ]
        victim_frms = [FoolingRateMetric() for _ in victim_model_names]

    # Run attack over the dataset (100 images by default)
    for i, (x, y, _) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Adversarial images are created here
        advs = attacker(x, y)

        # Track accuracy
        cln_outs = model(normalize(x))
        adv_outs = model(normalize(advs))
        frm.update(y, cln_outs, adv_outs)

        # Save first batch of adversarial examples
        if i == 0:
            saved_imgs = advs.detach().cpu().mul(255).to(torch.uint8)
            img_grid = tv.utils.make_grid(saved_imgs, nrow=4)
            tv.io.write_png(img_grid, 'adv_batch_0.png')

        # Track transfer fooling rates if victim models are provided
        if victim_model_names:
            for _, (vmodel, vfrm) in enumerate(zip(victim_models, victim_frms)):
                v_cln_outs = vmodel(normalize(x))
                v_adv_outs = vmodel(normalize(advs))
                vfrm.update(y, v_cln_outs, v_adv_outs)

    # Print results
    cln_acc, adv_acc, fr = frm.compute()
    print(f'Surrogate ({model_name}): {cln_acc=:.2%}, {adv_acc=:.2%} ({fr=:.2%})')

    if victim_model_names:
        for vmodel, vfrm in zip(victim_models, victim_frms):
            vcln_acc, vadv_acc, vfr = vfrm.compute()
            print(
                f'Victim ({vmodel.model_name}): cln_acc={vcln_acc:.2%}, '
                f'adv_acc={vadv_acc:.2%} (fr={vfr:.2%})'
            )
