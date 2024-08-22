from typing import Any, Callable, Self

import torch
import torch.nn as nn


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


class AttackModel:
    """A wrapper class for a pretrained model used for adversarial attacks.

    Intended to be instantiated with
    `AttackModel.from_pretrained(pretrained_model_name)` from either
    `torchvision.models` or `timm`. The model is loaded and attributes including
    `transform` and `normalize` are attached based on the model's configuration.

    Attributes:
        model_name (str): The name of the model.
        device (torch.device): The device on which the model is loaded.
        model (nn.Module): The pretrained model itself.
        transform (Callable): The transformation function applied to input images.
        normalize (Callable): The normalization function applied to input images.

    Example:
        >>> model = AttackModel.from_pretrained('resnet50', device='cuda')
        >>> model
        AttackModel(model_name=resnet50, device=cuda, transform=Compose(...), normalize=Normalize(...))
        >>> model.transform
        Compose(
            Resize(size=[256], interpolation=bilinear, max_size=None, antialias=True)
            CenterCrop(size=(224, 224))
            ToTensor()
        )
        >>> model.normalize
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        >>> model.model
        ResNet(
            (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            ...
        )
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        model: nn.Module,
        transform: Callable,
        normalize: Callable,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = model
        self.transform = transform
        self.normalize = normalize

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device,
        from_timm: bool = False,
    ) -> Self:
        """
        Loads a pretrained model and initializes an AttackModel instance.

        Args:
            model_name: The name of the model to load.
            device: The device on which to load the model.
            from_timm: Whether to load the model from timm. Defaults to False.

        Returns:
            AttackModel: An instance of AttackModel initialized with pretrained model.
        """

        import torchvision.transforms as t

        if not from_timm:
            try:
                import torchvision.models as tv_models

                model = tv_models.get_model(name=model_name, weights='DEFAULT')

                # Resolve transforms from vision model weights
                weight_id = str(tv_models.get_model_weights(name=model_name)['DEFAULT'])
                cfg = tv_models.get_weight(weight_id).transforms()

                # torchvision/transforms/_presets.py::ImageClassification
                # Manually construct separated transform and normalize
                transform = t.Compose(
                    [
                        t.Resize(
                            cfg.resize_size,
                            interpolation=cfg.interpolation,
                            antialias=cfg.antialias,
                        ),
                        t.CenterCrop(cfg.crop_size),
                        t.ToTensor(),
                    ]
                )
                normalize = t.Normalize(mean=cfg.mean, std=cfg.std)

            except ValueError:
                print('Model not found in torchvision.models, falling back to timm.')
                from_timm = True

        else:
            import timm

            model = timm.create_model(model_name, pretrained=True)
            cfg = timm.data.resolve_data_config(model.pretrained_cfg)

            # Construct normalization
            normalize = t.Normalize(mean=cfg['mean'], std=cfg['std'])

            # Create a transform based on the model pretrained cfg
            transform = timm.data.create_transform(**cfg, is_training=False)
            # Remove the Normalize from composed transform if there is one
            transform.transforms = [
                tr for tr in transform.transforms if not isinstance(tr, t.Normalize)
            ]

        model = model.to(device).eval()
        return cls(model_name, device, model, transform, normalize)

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(model_name={self.model_name}, device={self.device}, '
            f'transform={self.transform}, normalize={self.normalize})'
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
        model=model.model,
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
    print(
        f'Surrogate ({model_name}): {cln_acc:.2%} / {adv_acc:.2%} '
        f'(Fooling rate: {fr:.2%})'
    )

    if victim_model_names:
        for vmodel, vfrm in zip(victim_models, victim_frms):
            vcln_acc, vadv_acc, vfr = vfrm.compute()
            print(
                f'Victim ({vmodel.model_name}): {vcln_acc:.2%} / {vadv_acc:.2%} '
                f'(Fooling rate: {vfr:.2%})'
            )
