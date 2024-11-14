from lettuceSee import util
import unittest
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


class TestCropRegion(unittest.TestCase):
    def test_odd_shape(self):
        test_im = np.arange(0, 25).reshape((5, 5))
        shape = (3, 3)
        centre = (2, 2)
        crop = util.crop_region(test_im, centre, shape)
        assert(crop.shape == shape)
        assert(test_im[centre[1], centre[0]] == crop[shape[0] // 2, shape[1] // 2])

    def test_even_shape(self):
        test_im = np.arange(0, 25).reshape((5, 5))
        shape = (2, 2)
        centre = (2, 2)
        crop = util.crop_region(test_im, centre, shape)
        assert(crop.shape == shape)
        assert (test_im[centre[1], centre[0]] == crop[shape[0] // 2, shape[1] // 2])

