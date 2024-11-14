#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 11-11-2024

Functions for creating distance maps of masked images where each pixel gets assigned a value based on their proximity
to the nearest seed.
"""

import numpy as np
import scipy.ndimage as ndi


def coord_dist_map(
        mask: np.ndarray, seed_coords: list[tuple[int, int]]
) -> np.ndarray:
    """ Creates a distance map from seeds given as coordinates, bound by a mask

    :param mask: The binary mask that gives the boundaries of the distance map
    :param seed_coords: The list of coordinates
    :return: A distance map
    """
    seed_map = np.zeros_like(mask)
    for x, y in seed_coords:
        seed_map[y, x] = 1
    dist_map = im_dist_map(mask == 1, seed_map)
    return dist_map


def im_dist_map(mask: np.ndarray, seed_map: np.ndarray[bool]) -> np.ndarray:
    """ Creates a distance map from seeds derived from an input image, bound by a mask

    :param mask: The binary mask that gives the boundaries of the distance map
    :param seed_map: Binary image containing the seeds as 1 or True
    :return: A distance map
    """
    seed_map = np.invert(seed_map)
    seed_map[mask == 0] = 1

    if (seed_map == 0).sum() == 0:
        dist_map = np.zeros_like(mask)
    else:
        dist_map = ndi.distance_transform_edt(seed_map)

    dist_map[mask == 0] = 0
    return dist_map


def centre_seed_dist(flip_mask: np.ndarray[bool], non_flip_mask: np.ndarray[bool], seed_map: np.ndarray[bool]) -> \
        np.ndarray:
    """ Creates a dist map from a line bisecting a mask and transforms it into a continuous distance map

    :param flip_mask: Mask of the area of the distance map that will be flipped. The pixel that is furthest from the
        bisecting line in seed_map will be 0 in the output map
    :param non_flip_mask: Mask of the area of the distance map that will not be flipped
    :param seed_map: The seed map containing the line that bisects the mask into flip_mask and non_flip_mask, should
        have overlap with both flip_mask and non_flip_mask
    :return: A distance map
    """
    dist_map_f = im_dist_map(flip_mask, seed_map)
    dist_map_f_inverted = abs(dist_map_f - np.max(dist_map_f))
    dist_map_f_inverted[flip_mask == 0] = 0

    dist_map_nf = im_dist_map(non_flip_mask, seed_map)
    dist_map_nf += np.max(dist_map_f_inverted)
    dist_map_nf[non_flip_mask == 0] = 0

    dist_map = dist_map_f_inverted + dist_map_nf
    return dist_map
