#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 11/10/2023

Utility functions for image analysis
"""
import numpy as np


def crop_region(
        image: np.ndarray, centre: int, shape: tuple[int, int]
) -> np.ndarray:
    """ Crops an image area of specified width and height around a central point

    :param image: matrix representing the image
    :param centre: x and y coordinate of the centre as integers
    :param shape: contains the height and width of the subregion in pixels as
        integers
    :return: The cropped region of the original image
    """
    if image.ndim == 2:
        crop = image[
               centre[1] - shape[0] // 2: centre[1] + shape[0] // 2,
               centre[0] - shape[1] // 2: centre[0] + shape[1] // 2
               ]
    else:
        crop = image[
               centre[1] - shape[0] // 2: centre[1] + shape[0] // 2,
               centre[0] - shape[1] // 2: centre[0] + shape[1] // 2,
               :
               ]
    return crop


def read_fimg(filename: str) -> np.ndarray:
    """ Turns an FIMG value into a normalized file with data between 0 and 1

    :param filename: name of the file that is to be opened
    :return: 2D array representing the fimg image
    """
    image = np.fromfile(filename, np.dtype("float32"))
    image = image[2:]
    image = np.reshape(image, newshape=(1024, 1360))
    image[image < 0] = 0
    return image


def increase_contrast(im_channel: np.ndarray) -> np.ndarray:
    """ Takes a 2d array and makes its values range from 0 to 1

    :param im_channel: numpy array with 2 dimensions
    :return: input channel scaled from 0 to 1.
    """
    ch_min = im_channel.min()
    ch_max = im_channel.max()
    if ch_max == ch_min:
        raise ValueError("All values in the image are identical.")
    out = (im_channel - ch_min) / (ch_max - ch_min)
    return out


def multichannel_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Takes an image and applies a mask to every channel

    :param image: 3 dimensional array representing an image
    :param mask: 2d binary mask
    :return: masked input image
    """
    mask = mask.astype(image.dtype)
    image = image.copy()
    image[:, :, 0] *= mask
    image[:, :, 1] *= mask
    image[:, :, 2] *= mask
    return image


def paint_col(
        image: np.ndarray, mask: np.ndarray, color_tuple: tuple[int, int, int]
) -> np.ndarray:
    """ Makes masked area the specified color

    :param image: 3d array representing an RGB image
    :param mask: 2d binary mask
    :param color_tuple: contains the values in integer of the R, G and B
        channel that you want to paint
    :return: same as input image but with the masked area painted in
        the specified color
    """
    image = image.copy()
    image[:, :, 0][mask == 0] = color_tuple[0]
    image[:, :, 1][mask == 0] = color_tuple[1]
    image[:, :, 2][mask == 0] = color_tuple[2]
    return image


def merge_masks(bg_mask: np.ndarray, pheno_mask: np.ndarray) -> np.ndarray:
    """ Merges 2 binary masks into one mask, where phenotype has high values

    :param bg_mask: 2D mask with background as 0 and foreground as
        1
    :param pheno_mask: 2D mask phenotype marked as 1 and everything
        else as 0
    :return: 2D mask with background marked as 0, foreground as 1 and
        phenotype area as 2
    """
    substep = bg_mask.astype(int) + pheno_mask.astype(int)
    comb_mask = np.zeros_like(bg_mask)
    comb_mask[substep == 1] = 2
    comb_mask[substep == 2] = 1
    comb_mask[bg_mask == 0] = 0
    return comb_mask


def threshold_between(
        image: np.ndarray, x_low: float | int = None,
        x_high: float | int = None, y_low: float | int = None,
        y_high: float | int = None, z_low: float | int = None,
        z_high: float | int = None, and_mask: bool = True
) -> np.ndarray:
    """ Thresholds an image array for being between two values for each channels

    :param image: 3d matrix representing the image
    :param x_low: low boundary for channel 1. Defaults to minimum of channel 1.
    :param x_high: high boundary for channel 1. Defaults to maximum of channel
        1.
    :param y_low: low boundary for channel 2. Defaults to minimum of channel 2.
    :param y_high: high boundary for channel 2. Defaults to maximum of channel
        2.
    :param z_low: low boundary for channel 3. Defaults to minimum of channel 3.
    :param z_high: high boundary for channel 3. Defaults to maximum of channel
        3.
    :param and_mask: if true returned mask is only true when all
        thresholds apply. If false returned mask is true if at least one of the
        thresholds apply.
    :return: binary mask
    """
    # channel x thresholding
    if not x_low:
        x_low = image[:, :, 0].min()
    if not x_high:
        x_high = image[:, :, 0].max()
    x_mask = (image[:, :, 0] >= x_low) & (image[:, :, 0] <= x_high)
    # channel y thresholding
    if not y_low:
        y_low = image[:, :, 1].min()
    if not y_high:
        y_high = image[:, :, 1].max()
    y_mask = (image[:, :, 1] >= y_low) & (image[:, :, 1] <= y_high)
    # channel z thresholding
    if not z_low:
        z_low = image[:, :, 2].min()
    if not z_high:
        z_high = image[:, :, 2].max()
    z_mask = (image[:, :, 2] >= z_low) & (image[:, :, 2] <= z_high)
    if and_mask:
        comp_mask = x_mask & y_mask & z_mask
    else:
        comp_mask = x_mask | y_mask | z_mask
    return comp_mask
