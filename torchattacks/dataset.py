import csv
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

__all__ = ["NIPSLoader", "IMAGENET_TRANSFORM"]

IMAGENET_TRANSFORM = nn.Sequential(
    transforms.Resize([256]),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
)


class NIPSDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        pairs_path: str,
    ) -> None:
        """The NIPS 2017 Adversarial Learning Challenge dataset (derived from ImageNet).

        https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set

        Dataset folder should contain the images in the following format:

        ```
        data/nips2017/
        |-- images/            <-- image_root
        |-- images.csv         <-- pairs_path
        `-- categories.csv
        ```

        Images from the dataset are loaded into `torch.Tensor` within the range [0, 1].
        It is your job to perform the necessary preprocessing normalizations before
        passing the images to your model or to attacks.

        Args:
            image_root: Path to the folder containing the images.
            pairs_path: Path to the csv file containing the image names and labels.
        """

        super().__init__()

        self.image_root = image_root
        self.pairs_path = pairs_path

        image_name, image_label = [], []

        with open(self.pairs_path) as pairs_csv:
            reader = csv.reader(pairs_csv)

            # Skip header
            _ = next(reader)

            for r in reader:
                image_name.append(r[0])
                image_label.append(r[6])

        self.names = image_name
        self.labels = image_label

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, int]:
        name = self.names[index]
        label = int(self.labels[index]) - 1

        image = Image.open(f"{self.image_root}/{name}.png").convert("RGB")
        image = np.array(image, dtype=np.uint8)
        image = torch.from_numpy(image).permute((2, 0, 1)).contiguous().float().div(255)
        # image = self.transforms(image) if self.transforms else image
        return name, image, label


class NIPSLoader(DataLoader):
    def __init__(
        self,
        path: str | None = "data/nips2017",
        image_root: str | None = None,
        pairs_path: str | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        # Specifing a custom image root directory is useful when evaluating
        # transferability on a generated adversarial examples folder
        self.image_root = f"{path}/images" if image_root is None else image_root
        self.pairs_path = f"{path}/images.csv" if pairs_path is None else pairs_path
        self.dataset = NIPSDataset(
            image_root=self.image_root,
            pairs_path=self.pairs_path,
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
