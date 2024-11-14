#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 10/10/2023

Contains functions for segmenting image arrays.
"""
import skimage as sk
import numpy as np
from scipy import signal
from . import util


def elevation_map(rgb_im: np.ndarray) -> np.ndarray:
    """ Creates an elevation map of an RGB image by summing the sobel outputs of each channel

    :param rgb_im: 3 dimensional array representing an RGB image
    :return: 2 dimensional array representing an edge map
    """
    compound_sobel = sk.filters.sobel(rgb_im)
    compound_sobel = compound_sobel[:, :, 0] + compound_sobel[:, :, 1] + \
                     compound_sobel[:, :, 2]
    elevation = sk.filters.sobel(compound_sobel)
    return elevation


def multichannel_threshold(
        multi_ch_im, x_th: float = 0.0, y_th: float = 0.0, z_th: float = 0.0,
        inverse: bool = False
) -> np.ndarray:
    """ Takes a three-channel image and returns a mask based on thresholds

    :param multi_ch_im: a numpy array representing an image with
        three color channels
    :param x_th: the threshold for the first channel, 0.0 by default
    :param y_th: the threshold for the second channel, 0.0 by default
    :param z_th: the threshold for the third channel, 0.0 by default
    :param inverse: if False pixels below the threshold are marked as 0,
        if True, pixels above the threshold are marked as 0.
    :return: the mask created based on the thresholds, 2D array
        same width and height as the input
    """
    mask = np.ones(multi_ch_im.shape[0:2])
    mask[multi_ch_im[:, :, 0] < x_th] = 0
    mask[multi_ch_im[:, :, 1] < y_th] = 0
    mask[multi_ch_im[:, :, 2] < z_th] = 0
    mask = mask.astype(int)
    if inverse:
        mask = np.invert(mask)
    return mask


def sh_markers(image: np.ndarray, distance: int = 10, bg_mod: float = 0.15,
               fg_mod: float = 0.2) -> np.ndarray:
    """ Creates marker image based on sobel histogram

    :param image: 3d array representing a 3 channel image
    :param distance: minimal distance between local maxima and minima
    :param bg_mod: modifier for histogram segmentation
    :param fg_mod: modifier for histogram segmentation
    :return: 2D marker image
    """
    if image.shape[2] == 4:
        image = sk.util.img_as_ubyte(sk.color.rgba2rgb(image))
    comp_sob = sk.filters.sobel(image)
    comp_sob = comp_sob[:, :, 0] + comp_sob[:, :, 1] + comp_sob[:, :, 2]
    values, bins = np.histogram(comp_sob, bins=100)
    max_i, _ = signal.find_peaks(values, distance=distance)
    max_bins = bins[max_i]
    min_i, _ = signal.find_peaks(-values, distance=distance)
    min_bins = bins[min_i]
    min_bins = min_bins[min_bins > max_bins[0]]
    markers = np.zeros_like(comp_sob)
    markers[
        comp_sob <= max_bins[0] + (bg_mod * (min_bins[0] - max_bins[0]))] = 1
    markers[
        comp_sob >= min_bins[0] + (fg_mod * (max_bins[1] - min_bins[0]))] = 2
    markers = markers.astype(int)
    return markers


def shw_segmentation(
        image: np.ndarray, distance: int = 10, bg_mod: float = 0.15,
        fg_mod: float = 0.2
) -> np.ndarray:
    """ Creates binary image through sobel + histogram thresholds + watershed

    :param image: array representing a 3d image
    :param distance: minimal distance between local maxima and minima
    :param bg_mod: modifier for histogram segmentation
    :param fg_mod: modifier for histogram segmentation
    :return: 2D mask for the image
    """
    if image.shape[2] == 4:
        image = sk.util.img_as_ubyte(sk.color.rgba2rgb(image))
    comp_sob = sk.filters.sobel(image)
    comp_sob = comp_sob[:, :, 0] + comp_sob[:, :, 1] + comp_sob[:, :, 2]
    elevation = sk.filters.sobel(comp_sob)
    values, bins = np.histogram(comp_sob, bins=100)
    max_i, _ = signal.find_peaks(values, distance=distance)
    max_bins = bins[max_i]
    min_i, _ = signal.find_peaks(-values, distance=distance)
    min_bins = bins[min_i]
    min_bins = min_bins[min_bins > max_bins[0]]
    markers = np.zeros_like(comp_sob)
    markers[
        comp_sob <= max_bins[0] + (bg_mod * (min_bins[0] - max_bins[0]))] = 1
    markers[
        comp_sob >= min_bins[0] + (fg_mod * (max_bins[1] - min_bins[0]))] = 2
    markers = markers.astype(int)
    mask = sk.segmentation.watershed(elevation, markers)
    return mask - 1


def barb_thresh(im_channel: np.ndarray, div: int = 3) -> float:
    """ Defines the threshold of an image channel based on its histogram

    :param im_channel: 2d array, meant to be hue channel of hsv or
        a channel of lab
    :param div: the divisor used at the end of the algorithm. A higher
        divisor will lead to a lower threshold
    :return: the threshold of the image channel that separates it into
        healthy and unhealthy tissue
    
    Based on method described in `(Barbedo, 2016) <https://doi.org/10.1007/s40858-016-0090-8>`_
    """
    values, bins = np.histogram(im_channel, bins=100)
    peak_i = np.argmax(values)
    val_max = values[peak_i]
    bin_max = bins[peak_i]
    if bin_max <= 0.4:
        bound = 0.2 * val_max
    else:
        bound = 0.5 * val_max
    ref_val = values[values > bound][-1]
    ref_i = np.where(values == ref_val)[0][0]
    ref_bin = bins[ref_i]
    thresh = 2 * ref_bin / div
    return thresh


def barb_hue(
        image: np.ndarray, bg_mask: np.ndarray = None, div: int = 3
) -> np.ndarray:
    """ Takes an image of plant tissue and segments into healthy and brown

    :param image: 3d array representing an rgb image
    :param bg_mask: 2d array to mask the background
    :param div: the divisor used at the end of the algorithm. A higher
        divisor will lead to a lower threshold
    :return: mask with background as 0, healthy tissue as 1 and
        brown tissue as 2

    Based on method described in `(Barbedo, 2016) <https://doi.org/10.1007/s40858-016-0090-8>`_
    """
    if bg_mask is not None:
        # Apply mask to rgb_im
        image = util.multichannel_mask(image, bg_mask)
    # Get hue channel and scale from 0 to 1
    hue = sk.color.rgb2hsv(image)[:, :, 0]
    hue_con = util.scale_zero_to_one(hue)
    hue_fg = hue_con[bg_mask == 1]
    # Healthy tissue masking
    thresh = barb_thresh(hue_fg, div)
    healthy_mask = (hue_con > thresh).astype(int)
    # Remove noise
    healthy_mask = sk.morphology.remove_small_holes(
        healthy_mask,
        area_threshold=(image.shape[0] + image.shape[1]) // 200
    )
    # Combine healthy and bg mask to get compound image
    bg_mask = bg_mask.astype(int)
    comp_mask = util.merge_masks(bg_mask, healthy_mask)
    return comp_mask


def canny_labs(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """ Separates objects trough canny edge detection and then labels the output

    :param image: 2d array representing an image
    :param mask: 2d binary mask
    :param sigma: the sigma used for the gaussian blur component of canny
        segmentation
    :return: labelled image
    """
    canny_f = sk.feature.canny(image, sigma=sigma)
    canny_f = sk.morphology.closing(
        canny_f, footprint=sk.morphology.disk(radius=min(image.shape) // 500)
    )
    canny_f = sk.morphology.skeletonize(canny_f)
    mask = mask.copy()
    mask[canny_f == 1] = 0
    labels = sk.measure.label(mask, connectivity=1)
    return sk.morphology.dilation(labels)


def centre_primary_label(
        lab_im: np.ndarray, edge_length: int = 200, bg_label: int = 0
) -> np.ndarray:
    """ returns the label of the largest object within a central square with ribs of edge_length

    :param lab_im: labelled image with only positive values and 0
    :param edge_length: height and width of the square used on the centre
    :param bg_label: the label number that will be considered background.
        This can not be chosen as the primary label.
    :return: primary label
    """
    centre = (lab_im.shape[0] // 2, lab_im.shape[1] // 2)
    crop = util.crop_region(lab_im, centre, (edge_length, edge_length))
    return np.argmax(np.bincount(crop[crop != bg_label].ravel()))


def central_ob(mask: np.ndarray, edge_length: int = 200) -> np.ndarray:
    """ Selects the largest object within a central square with ribs of edge_length

    :param mask: 2d array representing a background mask
    :param edge_length: height and width of the square considered the
        centre
    :return: 2d array representing the input mask with only the
        largest central object
    """
    label_image = sk.measure.label(mask)
    central_lab = centre_primary_label(label_image, edge_length)
    centre_mask = (label_image == central_lab).astype(int)
    return centre_mask


def canny_central_ob(
        image: np.ndarray, mask: np.ndarray, sigma: float, central_area: int = 200,
        allowed_deviation: tuple[tuple[int | float, int | float], ...] = ((0.1, 0.1), (0.25, 0.25), (0.25, 0.25))
) -> np.ndarray:
    """ Uses canny filter and color channel thresholding to take central object

    :param image: 3d array representing color image
    :param mask: 2d boolean array representing background mask
    :param sigma: sigma used for gaussian blur step of canny edge
        detection
    :param central_area: central area size
    :param allowed_deviation: tuple with nested tuple for each channel of an image. Each nested tuple contains the lower
        and upper deviation allowed for that channel
    :return: 2d binary mask of central object
    """
    bg_labs = sk.measure.label(mask)
    mask = bg_labs == centre_primary_label(bg_labs)
    canny_labelled = canny_labs(sk.color.rgb2gray(image), mask, sigma)
    prim_lab = centre_primary_label(canny_labelled, central_area)
    average_cols = sk.color.label2rgb(canny_labelled, image, kind="avg")
    average_cols = sk.color.rgb2hsv(average_cols)
    prim_area = util.multichannel_mask(average_cols, canny_labelled == prim_lab)
    ch_one = np.unique(prim_area[:, :, 0])[1]
    ch_two = np.unique(prim_area[:, :, 1])[1]
    ch_three = np.unique(prim_area[:, :, 2])[1]
    mask = util.threshold_between(
        image=average_cols,
        x_low=ch_one - allowed_deviation[0][0], x_high=ch_one + allowed_deviation[0][1],
        y_low=ch_two - allowed_deviation[1][0], y_high=ch_two + allowed_deviation[1][1],
        z_low=ch_three - allowed_deviation[2][0], z_high=ch_three + allowed_deviation[2][1]
    )
    return mask
