import math
from dataclasses import dataclass
from typing import Callable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn
from PIL import Image

try:
    import torchvision.transforms as t
    import torchvision.transforms.functional as f
except ImportError as e:
    raise ImportError('`torchvision` is required to craft necessary transforms.') from e


class TvTransform(nn.Module):
    """Reimplementation of `torchvision.transforms._presets.ImageClassification`.

    Note:
        We do not import directly from `torchvision.transforms._presets` as it is
        declared as private API and is subject to change without warning.

    Args:
        crop_size: The size of the center crop.
        resize_size: The size to resize the image to.
        mean: The mean values for normalization. Not used.
        std: The standard deviation values for normalization. Not used.
        interpolation: The interpolation mode for resizing.
        antialias: Whether to apply antialiasing during resizing.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        crop_size: list[int],
        resize_size: list[int],
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: f.InterpolationMode = f.InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.mean = mean  # not used
        self.std = std  # not used
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
            f'resize_size={self.resize_size}, '
            f'crop_size={self.crop_size}, '
            f'interpolation={self.interpolation}, '
            f'antialias={self.antialias})'
        )


@dataclass
class AttackModelMeta:  # type: ignore[no-any-unimported]
    """AttackModelMeta class for handling image preprocessing parameters.

    Note:
        This class is used internally to resolve the image preprocessing parameters
        from pretrained models in `timm` and `torchvision.models` automatically.

    Attributes:
        resize_size: The size to resize images to before cropping.
        crop_size: The final size of the image after cropping.
        interpolation: Resize interpolation. Defaults to `InterpolationMode.BILINEAR`.
        antialias: Whether to use antialiasing when resizing images. Defaults to True.
        mean: Mean values for image normalization across RGB channels. Defaults to
            ImageNet means (0.485, 0.456, 0.406).
        std: Standard deviation values for image normalization across RGB channels.
            Defaults to ImageNet standard deviations (0.229, 0.224, 0.225).
    """

    resize_size: int
    crop_size: int
    interpolation: f.InterpolationMode = f.InterpolationMode.BILINEAR  # type: ignore[no-any-unimported]
    antialias: bool = True
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    @classmethod
    def from_timm_pretrained_cfg(cls, cfg: dict) -> Self:
        from timm.data.transforms import str_to_interp_mode

        # Reference:
        # create_transform::https://github.com/huggingface/pytorch-image-models/blob/a49b02/timm/data/transforms_factory.py#L334
        # transforms_imagenet_eval::https://github.com/huggingface/pytorch-image-models/blob/a49b02/timm/data/transforms_factory.py#L247
        crop_size = (
            cfg['input_size'][-1]
            if isinstance(cfg['input_size'], (tuple, list))  # (3, 224, 224)
            else cfg['input_size']
        )
        resize_size = math.floor(crop_size / cfg['crop_pct'])
        return cls(
            resize_size=resize_size,
            crop_size=crop_size,
            interpolation=str_to_interp_mode(cfg['interpolation']),
            mean=cfg['mean'],
            std=cfg['std'],
        )

    @classmethod
    def from_tv_transforms(cls, cfg: TvTransform) -> Self:
        return cls(
            resize_size=cfg.resize_size[0],
            crop_size=cfg.crop_size[0],
            interpolation=cfg.interpolation,
            antialias=cfg.antialias,
            mean=cfg.mean,
            std=cfg.std,
        )

    @classmethod
    def from_preprocessings(
        cls,
        transform: Callable[[Image.Image | torch.Tensor], torch.Tensor],
        normalize: Callable[[torch.Tensor], torch.Tensor],
    ) -> Self:
        """
        Create AttackModelMeta from transform and normalize functions. If transform or
        normalize don't expose the expected attributes, return a default meta with zeros
        and defaults for interpolation, etc.
        """

        # early return default meta if we can't inspect the args
        if not (
            hasattr(transform, 'transforms')
            and hasattr(normalize, 'mean')
            and hasattr(normalize, 'std')
        ):
            # resize_size and crop_size default to 0; other fields use their defaults
            return cls(0, 0)

        crop_size: int | None = None
        resize_size: int | None = None
        interpolation = cls.interpolation
        antialias = cls.antialias

        for tfs in transform.transforms:
            if isinstance(tfs, t.CenterCrop):
                cs = tfs.size
                crop_size = cs[0] if isinstance(cs, (list, tuple)) else cs
            elif isinstance(tfs, t.Resize):
                rs = tfs.size
                resize_size = rs[0] if isinstance(rs, (list, tuple)) else rs
                interpolation = tfs.interpolation
                antialias = tfs.antialias

        # Fallback if transform lacked one of the required ops
        if crop_size is None or resize_size is None:
            return cls(0, 0)

        return cls(
            resize_size,
            crop_size,
            interpolation=interpolation,
            antialias=antialias,
            mean=tuple(normalize.mean),
            std=tuple(normalize.std),
        )


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
        meta: Model transform meta info. If not provided, will be inferred from
            `transform` and `normalize` functions. Defaults to None.

    Example:
        ```pycon
        >>> model = AttackModel.from_pretrained('resnet50')
        >>> model
        AttackModel(model_name=resnet50, transform=Compose(...), normalize=Normalize(...))
        >>> model.transform
        TvTransform(crop_size=[224], resize_size=[232], interpolation=InterpolationMode.BILINEAR)
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
        meta: AttackModelMeta | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.transform = transform
        self.normalize = normalize
        self.meta = meta or AttackModelMeta.from_preprocessings(transform, normalize)

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
            An instance of AttackModel initialized with pretrained model.
        """

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
            meta = AttackModelMeta.from_timm_pretrained_cfg(cfg)

            # Construct normalization
            normalize = t.Normalize(mean=cfg['mean'], std=cfg['std'])

            # Create a transform based on the model pretrained cfg
            transform = timm.data.create_transform(**cfg, is_training=False)
            # Remove the Normalize from composed transform if there is one
            transform.transforms = [
                tr for tr in transform.transforms if not isinstance(tr, t.Normalize)
            ]

            return cls(model_name, model, transform, normalize, meta)

        # If the model is not specified to be load from timm, try loading from
        # `torchvision.models` first, then fall back to timm if the model is not found.
        try:
            import torchvision.models as tv_models

            model = tv_models.get_model(model_name, weights='DEFAULT').eval()

            # Resolve transforms from vision model weights
            weight_id = str(tv_models.get_model_weights(name=model_name)['DEFAULT'])
            cfg = tv_models.get_weight(weight_id).transforms()
            meta = AttackModelMeta.from_tv_transforms(cfg)

            # Manually construct separated transform and normalize
            transform = TvTransform(
                crop_size=cfg.crop_size,
                resize_size=cfg.resize_size,
                interpolation=cfg.interpolation,
                antialias=cfg.antialias,
            )
            normalize = t.Normalize(mean=cfg.mean, std=cfg.std)

            return cls(model_name, model, transform, normalize, meta)

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
            The AttackModel instance with the updated device.
        """

        self.model = self.model.to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: torch.Tensor = self.model(x)
        return outs

    def create_relative_transform(
        self, other: Self
    ) -> Callable[[Image.Image | torch.Tensor], torch.Tensor]:
        """Create relative transform function between two AttackModel instances.

        Compose minimal, i.e., just enough, transforms, by not introducing unnecessary
        resizes if the input image size is already the same as the default size of the
        other model's input. Ensures that no unnecessary resizing is performed that may
        affect the effectiveness of the adversarial perturbation generated.

        Note:
            Relative means that we assume **the input has already been transformed (most
            often resized and center cropped) with transforms defined as in the `other`
            AttackModel**. We then, in this case, would not require applying the same
            transform again if the final input size is the same. The created transform
            accepts inputs of both `PIL.Image` and `torch.Tensor`.

        Example:
            ```pycon
            >>> res50 = AttackModel.from_pretrained('resnet50')
            >>> incv3 = AttackModel.from_pretrained('inception_v3')
            >>> transform = incv3.create_relative_transform(res50)
            >>> transform
            Compose(
                Resize(size=299, interpolation=bilinear, antialias=True)
                MaybePIlToTensor()
            )
            ```

        Args:
            other: The other AttackModel instance.

        Returns:
            The created relative transform.
        """

        tfl = []

        class MaybePIlToTensor:
            def __call__(self, x: Image.Image | torch.Tensor) -> torch.Tensor:
                if not isinstance(x, torch.Tensor):
                    x = f.pil_to_tensor(x)
                x = f.convert_image_dtype(x, torch.float)
                return x  # type: ignore[return-value]

            def __repr__(self) -> str:
                return f'{self.__class__.__name__}()'

        # Ignore resize size, perform size transform only if crop size is different
        if self.meta.crop_size != other.meta.crop_size:
            tfl += [
                t.Resize(
                    self.meta.crop_size,
                    interpolation=self.meta.interpolation,
                    antialias=self.meta.antialias,
                )
            ]
        tfl += [MaybePIlToTensor()]
        return t.Compose(tfl)  # type: ignore[no-any-return]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'model_name={self.model_name}, '
            f'transform={self.transform}, '
            f'normalize={self.normalize}, '
            f'meta={self.meta})'
        )
