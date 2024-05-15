import numpy as np
from typing import Tuple

def index_from_pdf(pdf_im):
    prob = np.copy(pdf_im)
    # Normalize values to create a pdf with sum = 1
    prob = prob.ravel() / np.sum(prob)
    # Convert into a 1D pdf
    choices = np.prod(pdf_im.shape)
    index = np.random.choice(choices, size=1, p=prob)
    # Recover 2D shape
    coordinates = np.unravel_index(index, shape=pdf_im.shape)
    # Extract index
    indexh = coordinates[0][0]
    indexw = coordinates[1][0]
    return indexh, indexw

def sampling_pdf(y, pdf, height, width):
    h, w = y.shape[0], y.shape[1]
    #h, w = y.shape[0], y.shape[1]
    if pdf == 1:
        indexw = np.random.randint(np.floor(width // 2), \
                                   w - np.floor(width // 2))
        indexh = np.random.randint(np.floor(height // 2), \
                                   h - np.floor(height // 2))
    else:
        # Assign pdf values to foreground
        pdf_im = np.ones(y.shape, dtype=np.float32)
        pdf_im[y > 0] = pdf
        # crop to fix patch size
        pdf_im = pdf_im[np.int16(np.floor(height // 2)):-np.int16(np.floor(height // 2)), \
                 np.int16(np.floor(width // 2)):-np.int16(np.floor(width // 2))]
        indexh, indexw = index_from_pdf(pdf_im)
        indexw = indexw + np.int16(np.floor(width // 2))
        indexh = indexh + np.int16(np.floor(height // 2))

    return indexh, indexw

    
def sample_crops(mask, label_weight=3000, patch_size = [256, 256]):
    # Sample training images to match the desired distribution from to min and max of the source distribution

    # Random draw, check target distribution 
    indexh, indexw = sampling_pdf(mask, label_weight, patch_size[0], patch_size[1])
    lr = indexh - np.floor(patch_size[0] // 2)
    lr = lr.astype(np.int16)
    ur = indexh + np.round(patch_size[0] // 2)
    ur = ur.astype(np.int16)

    lc = indexw - np.floor(patch_size[1] // 2)
    lc = lc.astype(np.int16)
    uc = indexw + np.round(patch_size[1] // 2)
    uc = uc.astype(np.int16)

    coord = (lr, ur, lc, uc)
    #image_patch = image[lr:ur, lc:uc]
    #mask_patch = np.expand_dims(mask_patch, axis=0)
    #image_patch = np.expand_dims(image_patch, axis=0)
    #self.loaded_masks_patches[i] = transforms.ToTensor()(mask_patch)
    #self.loaded_input_patches[i] = transforms.ToTensor()(image_patch)
    return coord


def normalize(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Normalize an image with given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    mean : float, optional
        Mean value for normalization, by default 0.0.
    std : float, optional
        Standard deviation value for normalization, by default 1.0.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    return (image - mean) / std


def denormalize(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Denormalize an image with given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    mean : float, optional
        Mean value for normalization, by default 0.0.
    std : float, optional
        Standard deviation value for normalization, by default 1.0.

    Returns
    -------
    np.ndarray
        Denormalized array.
    """
    return image * std + mean


def _flip_and_rotate(
    image: np.ndarray, rotate_state: int, flip_state: int
) -> np.ndarray:
    """
    Apply the given number of 90 degrees rotations and flip to an array.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    rotate_state : int
        Number of 90 degree rotations to apply.
    flip_state : int
        0 or 1, whether to flip the array or not.

    Returns
    -------
    np.ndarray
        Flipped and rotated array.
    """
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated
    return flipped.copy()


def augment_batch(
    patch: np.ndarray,
    target: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    Apply augmentation function to patches and masks.

    Parameters
    ----------
    patch : np.ndarray
        Array containing single image or patch, 2D or 3D with masked pixels.
    original_image : np.ndarray
        Array containing original image or patch, 2D or 3D.
    mask : np.ndarray
        Array containing only masked pixels, 2D or 3D.
    seed : int, optional
        Seed for random number generator, controls the rotation and falipping.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Tuple of augmented arrays.
    """
    rng = np.random.default_rng(seed=seed)
    rotate_state = rng.integers(0, 4)
    flip_state = rng.integers(0, 2)
    return (
        _flip_and_rotate(patch, rotate_state, flip_state),
        _flip_and_rotate(target, rotate_state, flip_state),
    )