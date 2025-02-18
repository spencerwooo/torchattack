import os

import torch
from PIL import Image


def save_image_batch(
    imgs: torch.Tensor,
    save_dir: str,
    filenames: list[str] | None = None,
    extension: str = 'png',
    kwargs: dict | None = None,
) -> None:
    """Losslessly (as lossless as possible) save a batch of images to disk.

    Args:
        imgs: The batch of images to save.
        save_dir: The directory to save the images (parent folder).
        filenames: The names of the images without their extensions. Defaults to None.
        extension: The extension of the images to save as. One of 'png', 'jpeg'. Defaults to "png".
        kwargs: Additional keyword arguments to pass to the image save function. Defaults to None.
    """

    if kwargs is None:
        kwargs = (
            {}
            # To best preserve perturbation effectiveness, we recommend saving as PNGs
            # See: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#png-saving
            if extension == 'png'
            # If saving as JPEGs, add additional arguments to ensure perturbation quality
            # See: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg-saving
            else {'quality': 100, 'subsampling': 0, 'keep_rgb': True}
        )

    assert extension in ['png', 'jpeg'], 'Extension must be either `png` or `jpeg`.'

    # Create the parent directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate random filenames if none are provided
    if filenames is None:
        filenames = _generate_filenames(len(imgs))

    assert imgs.dim() == 4, 'Input tensor must be 4D (BCHW)'
    assert imgs.size(0) == len(filenames), 'Batch size must match number of filenames'

    for x, name in zip(imgs, filenames):
        img = x.detach().cpu().numpy().transpose(1, 2, 0)
        img = Image.fromarray((img * 255).astype('uint8'))
        img.save(os.path.join(save_dir, f'{name}.{extension}'), **kwargs)


def _generate_filenames(num: int, name_len: int = 10) -> list[str]:
    """Generate a list of random filenames.

    Args:
        num: The number of filenames to generate.
        name_len: The length of each filename. Defaults to 10.

    Returns:
        A list of random filenames.
    """

    import random
    import string

    characters = string.ascii_letters + string.digits
    return [''.join(random.choices(characters, k=name_len)) for _ in range(num)]
