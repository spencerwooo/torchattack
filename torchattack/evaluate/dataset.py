import csv
import os
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
        image_csv: str,
        transform: Callable[[torch.Tensor | Image.Image], torch.Tensor] | None = None,
        max_samples: int | None = None,
        return_target_label: bool = False,
    ) -> None:
        """Initialize the NIPS 2017 Adversarial Learning Challenge dataset.

        Dataset folder should contain the images in the following format:

        ```
        data/nips2017/
        ├── images/            <-- image_root
        ├── images.csv         <-- image_csv
        └── categories.csv
        ```

        Images from the dataset are loaded into `torch.Tensor` within the range [0, 1].
        It is your job to perform the necessary preprocessing normalizations before
        passing the images to your model or to attacks.

        Args:
            image_root: Path to the folder containing the images.
            image_csv: Path to the csv file containing the image names and labels.
            transform: An optional transform to apply to the images. Defaults to None.
            max_samples: Maximum number of samples to load. Defaults to None.
        """

        super().__init__()

        self.image_root = image_root
        self.image_csv = image_csv
        self.transform = transform
        self.return_target_label = return_target_label

        # Load data from CSV file
        self.names, self.labels, self.target_labels = self._load_metadata(max_samples)

    def _load_metadata(
        self, max_samples: int | None = None
    ) -> tuple[list[str], list[str], list[str]]:
        """Load image filenames and labels (ground truth + target labels) from the CSV.

        Args:
            max_samples: Maximum number of samples to load. Defaults to None.

        Returns:
            Tuple of (filenames, class_labels, target_labels)
        """

        names, labels, target_labels = [], [], []

        with open(self.image_csv) as pairs_csv:
            reader = csv.reader(pairs_csv)
            next(reader)  # Skip header row

            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break  # Limit dataset size if requested
                names.append(row[0])
                labels.append(row[6])
                target_labels.append(row[7])

        return names, labels, target_labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor | Image.Image, int, str]
        | tuple[torch.Tensor | Image.Image, tuple[int, int], str]
    ):
        """Get an item from the dataset by index.

        Args:
            index: Index of the item to retrieve

        Returns:
            If return_target_label is False: (image, label, image_name)
            If return_target_label is True: (image, (label, target_label), image_name)
        """

        # Get metadata for this sample
        filename = self.names[index]
        label = int(self.labels[index]) - 1  # Convert to 0-indexed

        # Load and process the image
        image_path = os.path.join(self.image_root, f'{filename}.png')
        pil_image = Image.open(image_path).convert('RGB')
        image = self.transform(pil_image) if self.transform else pil_image

        if self.return_target_label:
            target_label = int(self.target_labels[index]) - 1  # Convert to 0-indexed
            return image, (label, target_label), filename
        else:
            return image, label, filename


class NIPSLoader(DataLoader):
    """A custom dataloader for the NIPS 2017 dataset.

    Args:
        root: Path to the root folder containing the images and CSV file. Defaults to None.
        image_root: Path to the folder containing the images. Defaults to None.
        image_csv: Path to the csv file containing the image names and labels. Defaults to None.
        transform: An optional transform to apply to the images. Defaults to None.
        batch_size: Batch size for the dataloader. Defaults to 1.
        max_samples: Maximum number of samples to load. Defaults to None.
        return_target_label: Whether to return the target label in addition to the label. Defaults to False.
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
        specifying `image_root` and `image_csv`, which is usually used for evaluating
        models on a generated adversarial examples directory.
    """

    def __init__(
        self,
        root: str | None = None,
        image_root: str | None = None,
        image_csv: str | None = None,
        transform: Callable[[torch.Tensor | Image.Image], torch.Tensor] | None = None,
        batch_size: int = 1,
        max_samples: int | None = None,
        return_target_label: bool = False,
        num_workers: int = 4,
        shuffle: bool = False,
    ):
        assert root is not None or (image_root is not None and image_csv is not None), (
            'Either `root` or both `image_root` and `image_csv` must be specified'
        )

        # Specifing a custom image root directory is useful when evaluating
        # transferability on a generated adversarial examples folder
        super().__init__(
            dataset=NIPSDataset(
                image_root=image_root if image_root else os.path.join(root, 'images'),  # type: ignore
                image_csv=image_csv if image_csv else os.path.join(root, 'images.csv'),  # type: ignore
                transform=transform,
                max_samples=max_samples,
                return_target_label=return_target_label,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
