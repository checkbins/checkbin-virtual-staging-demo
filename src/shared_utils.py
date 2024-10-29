import numpy as np
import cv2 
from typing import Tuple, Union, List
import gc

import numpy as np
from PIL import Image
import torch
from scipy.signal import fftconvolve

from src.palette import COLOR_MAPPING, COLOR_MAPPING_

def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img

def resize_dimensions(dimensions, target_size):
    """ 
    Resize PIL to target size while maintaining aspect ratio 
    If smaller than target size leave it as is
    """
    width, height = dimensions

    # Check if both dimensions are smaller than the target size
    if width < target_size and height < target_size:
        return dimensions

    # Determine the larger side
    if width > height:
        # Calculate the aspect ratio
        aspect_ratio = height / width
        # Resize dimensions
        return (target_size, int(target_size * aspect_ratio))
    else:
        # Calculate the aspect ratio
        aspect_ratio = width / height
        # Resize dimensions
        return (int(target_size * aspect_ratio), target_size)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def filter_items(
    colors_list: Union[List, np.ndarray],
    items_list: Union[List, np.ndarray],
    items_to_remove: Union[List, np.ndarray]
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Filters items and their corresponding colors from given lists, excluding
    specified items.

    Args:
        colors_list: A list or numpy array of colors corresponding to items.
        items_list: A list or numpy array of items.
        items_to_remove: A list or numpy array of items to be removed.

    Returns:
        A tuple of two lists or numpy arrays: filtered colors and filtered
        items.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items

def to_rgb(color: str) -> tuple:
    """Convert hex color to rgb.
    Args:
        color (str): hex color
    Returns:
        tuple: rgb color
    """
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


def map_colors(color: str) -> str:
    """Map color to hex value.
    Args:
        color (str): color name
    Returns:
        str: hex value
    """
    return COLOR_MAPPING[color]


def map_colors_rgb(color: tuple) -> str:
    return COLOR_MAPPING_RGB[color]


def convolution(mask: Image.Image, size=9) -> Image:
    """Method to blur the mask
    Args:
        mask (Image): masking image
        size (int, optional): size of the blur. Defaults to 9.
    Returns:
        Image: blurred mask
    """
    mask = np.array(mask.convert("L"))
    conv = np.ones((size, size)) / size**2
    mask_blended = fftconvolve(mask, conv, 'same')
    mask_blended = mask_blended.astype(np.uint8).copy()

    border = size

    # replace borders with original values
    mask_blended[:border, :] = mask[:border, :]
    mask_blended[-border:, :] = mask[-border:, :]
    mask_blended[:, :border] = mask[:, :border]
    mask_blended[:, -border:] = mask[:, -border:]

    return Image.fromarray(mask_blended).convert("L")


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def postprocess_image_masking(inpainted: Image, image: Image,
                              mask: Image) -> Image:
    """Method to postprocess the inpainted image
    Args:
        inpainted (Image): inpainted image
        image (Image): original image
        mask (Image): mask
    Returns:
        Image: inpainted image
    """
    final_inpainted = Image.composite(inpainted.convert("RGBA"),
                                      image.convert("RGBA"), mask)
    return final_inpainted.convert("RGB")


COLOR_NAMES = list(COLOR_MAPPING.keys())
COLOR_RGB = [to_rgb(k) for k in COLOR_MAPPING_.keys()] + [(0, 0, 0),
                                                          (255, 255, 255)]
INVERSE_COLORS = {v: to_rgb(k) for k, v in COLOR_MAPPING_.items()}
COLOR_MAPPING_RGB = {to_rgb(k): v for k, v in COLOR_MAPPING_.items()}

