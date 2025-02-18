import csv
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class NIPSDataset(Dataset):
    """The NIPS 2017 Adversarial Learning Challenge dataset (derived from ImageNet).

    <https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set>
    """

    def __init__(
        self,
        image_root: str,
        pairs_path: str,
        transform: Callable[[torch.Tensor | Image.Image], torch.Tensor] | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the NIPS 2017 Adversarial Learning Challenge dataset.

        Dataset folder should contain the images in the following format:

        ```
        data/nips2017/
        ├── images/            <-- image_root
        ├── images.csv         <-- pairs_path
        └── categories.csv
        ```

        Images from the dataset are loaded into `torch.Tensor` within the range [0, 1].
        It is your job to perform the necessary preprocessing normalizations before
        passing the images to your model or to attacks.

        Args:
            image_root: Path to the folder containing the images.
            pairs_path: Path to the csv file containing the image names and labels.
            transform: An optional transform to apply to the images. Defaults to None.
            max_samples: Maximum number of samples to load. Defaults to None.
        """

        super().__init__()

        self.image_root = image_root
        self.pairs_path = pairs_path
        self.transform = transform

        image_name, image_label = [], []

        with open(self.pairs_path) as pairs_csv:
            reader = csv.reader(pairs_csv)

            # Skip header
            _ = next(reader)

            for r in reader:
                image_name.append(r[0])
                image_label.append(r[6])

        if max_samples is not None:
            image_name = image_name[:max_samples]
            image_label = image_label[:max_samples]

        self.names = image_name
        self.labels = image_label

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Image.Image | torch.Tensor, int, str]:
        name = self.names[index]
        label = int(self.labels[index]) - 1

        pil_image = Image.open(f'{self.image_root}/{name}.png').convert('RGB')
        # np_image = np.array(pil_image, dtype=np.uint8)
        # image = torch.from_numpy(np_image).permute((2, 0, 1)).contiguous().float().div(255)
        image = self.transform(pil_image) if self.transform else pil_image
        return image, label, name


class NIPSLoader(DataLoader):
    """A custom dataloader for the NIPS 2017 dataset.

    Args:
        root: Path to the root folder containing the images and CSV file. Defaults to None.
        image_root: Path to the folder containing the images. Defaults to None.
        pairs_path: Path to the csv file containing the image names and labels. Defaults to None.
        transform: An optional transform to apply to the images. Defaults to None.
        batch_size: Batch size for the dataloader. Defaults to 1.
        max_samples: Maximum number of samples to load. Defaults to None.
        num_workers: Number of workers for the dataloader. Defaults to 4.
        shuffle: Whether to shuffle the dataset. Defaults to False.

    Example:
        The dataloader reads image and label pairs (CSV file) from `{path}/images.csv`
        by default, and loads the images from `{path}/images/`.

        ```pycon
        >>> from torchvision.transforms import transforms
        >>> from torchattack.evaluate import NIPSLoader
        >>> transform = transforms.Compose([transforms.Resize([224]), transforms.ToTensor()])
        >>> dataloader = NIPSLoader(
        >>>     root="data/nips2017", transform=transform, batch_size=16, max_samples=100
        >>> )
        >>> x, y, fname = next(iter(dataloader))
        ```

        You can specify a custom image root directory and CSV file location by
        specifying `image_root` and `pairs_path`, which is usually used for evaluating
        models on a generated adversarial examples directory.
    """

    def __init__(
        self,
        root: str | None = None,
        image_root: str | None = None,
        pairs_path: str | None = None,
        transform: Callable[[torch.Tensor | Image.Image], torch.Tensor] | None = None,
        batch_size: int = 1,
        max_samples: int | None = None,
        num_workers: int = 4,
        shuffle: bool = False,
    ):
        assert root is not None or (
            image_root is not None and pairs_path is not None
        ), 'Either `root` or both `image_root` and `pairs_path` must be specified'

        # Specifing a custom image root directory is useful when evaluating
        # transferability on a generated adversarial examples folder
        super().__init__(
            dataset=NIPSDataset(
                image_root=image_root if image_root else f'{root}/images',
                pairs_path=pairs_path if pairs_path else f'{root}/images.csv',
                transform=transform,
                max_samples=max_samples,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
