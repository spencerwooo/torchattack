from typing import Callable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn
from PIL import Image


class AttackModel:
    """A wrapper class for a pretrained model used for adversarial attacks.

    Intended to be instantiated with `AttackModel.from_pretrained(<MODEL_NAME>)` from
    either `torchvision.models` or `timm`. The model is loaded and attributes including
    `model_name`, `transform`, and `normalize` are attached based on the model's config.

    Args:
        model_name: The name of the model.
        model: The pretrained model itself.
        transform: The transformation function applied to input images.
        normalize: The normalization function applied to input images.

    Example:
        ```pycon
        >>> model = AttackModel.from_pretrained('resnet50')
        >>> model
        AttackModel(model_name=resnet50, transform=Compose(...), normalize=Normalize(...))
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
        ```
    """

    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        transform: Callable[[Image.Image | torch.Tensor], torch.Tensor],
        normalize: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.transform = transform
        self.normalize = normalize

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        from_timm: bool = False,
    ) -> Self:
        """
        Loads a pretrained model and initializes an AttackModel instance.

        Args:
            model_name: The name of the model to load. Accept specifying the model from
                `timm` or `torchvision.models` by prefixing the model name with `timm/`
                or `tv/`. Takes precedence over the `from_timm` flag.
            from_timm: Explicitly specifying to load the model from timm or torchvision.
                Priority lower than argument `model_name`. Defaults to False.

        Returns:
            AttackModel: An instance of AttackModel initialized with pretrained model.
        """

        import torchvision.transforms as t

        # Accept `timm/<model_name>` or `tv/<model_name>` as model_name,
        # which takes precedence over the `from_timm` flag.
        if model_name.startswith('timm/'):
            model_name, from_timm = model_name[5:], True
        elif model_name.startswith('tv/'):
            model_name, from_timm = model_name[3:], False

        # Load the model from timm if specified
        if from_timm:
            import timm

            model = timm.create_model(model_name, pretrained=True).eval()
            cfg = timm.data.resolve_data_config(model.pretrained_cfg)

            # Construct normalization
            normalize = t.Normalize(mean=cfg['mean'], std=cfg['std'])

            # Create a transform based on the model pretrained cfg
            transform = timm.data.create_transform(**cfg, is_training=False)
            # Remove the Normalize from composed transform if there is one
            transform.transforms = [
                tr for tr in transform.transforms if not isinstance(tr, t.Normalize)
            ]

            return cls(model_name, model, transform, normalize)

        # If the model is not specified to be load from timm, try loading from
        # `torchvision.models` first, then fall back to timm if the model is not found.
        try:
            import torchvision.models as tv_models
            import torchvision.transforms.functional as f

            model = tv_models.get_model(model_name, weights='DEFAULT').eval()

            # Resolve transforms from vision model weights
            weight_id = str(tv_models.get_model_weights(name=model_name)['DEFAULT'])
            cfg = tv_models.get_weight(weight_id).transforms()

            # Source: torchvision/transforms/_presets.py::ImageClassification
            # We do not import directly from torchvision.transforms._presets as it is
            # declared private API and subject to change without warning.
            class TvTransform(nn.Module):
                def __init__(  # type: ignore[no-any-unimported]
                    self,
                    crop_size: list[int],
                    resize_size: list[int],
                    interpolation: f.InterpolationMode = f.InterpolationMode.BILINEAR,
                    antialias: bool | None = True,
                ) -> None:
                    super().__init__()
                    self.crop_size = crop_size
                    self.resize_size = resize_size
                    self.interpolation = interpolation
                    self.antialias = antialias

                def forward(self, x: Image.Image | torch.Tensor) -> torch.Tensor:
                    x = f.resize(
                        x,
                        self.resize_size,
                        interpolation=self.interpolation,
                        antialias=self.antialias,
                    )
                    x = f.center_crop(x, self.crop_size)
                    if not isinstance(x, torch.Tensor):
                        x = f.pil_to_tensor(x)
                    x = f.convert_image_dtype(x, torch.float)
                    return x  # type: ignore[return-value]

                def __repr__(self) -> str:
                    return (
                        f'{self.__class__.__name__}('
                        f'crop_size={self.crop_size}, '
                        f'resize_size={self.resize_size}, '
                        f'interpolation={self.interpolation})'
                    )

            # Manually construct separated transform and normalize
            transform = TvTransform(
                crop_size=cfg.crop_size,
                resize_size=cfg.resize_size,
                interpolation=cfg.interpolation,
                antialias=cfg.antialias,
            )
            normalize = t.Normalize(mean=cfg.mean, std=cfg.std)

            return cls(model_name, model, transform, normalize)

        except ValueError:
            from warnings import warn

            warn(
                f'model `{model_name}` not found in torchvision.models, '
                'falling back to loading weights from timm.',
                category=UserWarning,
                stacklevel=2,
            )
            return cls.from_pretrained(model_name, from_timm=True)

    def to(self, device: torch.device) -> Self:
        """Move the model to the specified device and update the device attribute.

        Args:
            device: The device to move the model to.

        Returns:
            AttackModel: The AttackModel instance with the updated device.
        """

        self.model = self.model.to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: torch.Tensor = self.model(x)
        return outs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'model_name={self.model_name}, '
            f'device={self.device}, '
            f'transform={self.transform}, '
            f'normalize={self.normalize})'
        )
