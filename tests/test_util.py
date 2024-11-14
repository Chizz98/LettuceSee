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
        assert (crop.shape == shape)
        assert (test_im[centre[1], centre[0]] == crop[shape[0] // 2, shape[1] // 2])

    def test_even_shape(self):
        test_im = np.arange(0, 25).reshape((5, 5))
        shape = (2, 2)
        centre = (2, 2)
        crop = util.crop_region(test_im, centre, shape)
        assert (crop.shape == shape)
        assert (test_im[centre[1], centre[0]] == crop[shape[0] // 2, shape[1] // 2])


class TestScaleZeroToOne(unittest.TestCase):
    def test_scaling(self):
        test_im = np.arange(0, 25).reshape((5, 5))
        scaled = util.scale_zero_to_one(test_im)
        assert (np.sum(scaled) == 12.5)
        assert (np.min(scaled) == 0)
        assert (np.max(scaled) == 1)

    def test_negative(self):
        test_im = np.arange(-5, 20).reshape((5, 5))
        scaled = util.scale_zero_to_one(test_im)
        assert (np.sum(scaled) == 12.5)
        assert (np.min(scaled) == 0)
        assert (np.max(scaled) == 1)


class TestMultichannelMask(unittest.TestCase):
    def test_masking(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = np.ones((3, 3))
        mask = np.pad(mask, 1)
        masked = util.multichannel_mask(test_im, mask)
        assert (np.array_equal(test_im[1:-1, 1:-1], masked[1:-1, 1:-1]))
        assert (np.sum(masked[mask == 0]) == 0)


class TestPaintCol(unittest.TestCase):
    def test_white(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = np.ones((3, 3))
        mask = np.pad(mask, 1)
        masked = util.paint_col(test_im, mask, (255, 255, 255))
        assert (np.array_equal(test_im[1:-1, 1:-1], masked[1:-1, 1:-1]))
        assert (np.average(masked[mask == 0]) == 255)

    def test_red(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = np.ones((3, 3))
        mask = np.pad(mask, 1)
        masked = util.paint_col(test_im, mask, (255, 0, 0))
        assert (np.array_equal(test_im[1:-1, 1:-1], masked[1:-1, 1:-1]))
        assert (np.average(masked[mask == 0][:, 0]) == 255)
        assert (np.average(masked[mask == 0][:, 1]) == 0)
        assert (np.average(masked[mask == 0][:, 2]) == 0)

    def test_green(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = np.ones((3, 3))
        mask = np.pad(mask, 1)
        masked = util.paint_col(test_im, mask, (0, 255, 0))
        assert (np.array_equal(test_im[1:-1, 1:-1], masked[1:-1, 1:-1]))
        assert (np.average(masked[mask == 0][:, 0]) == 0)
        assert (np.average(masked[mask == 0][:, 1]) == 255)
        assert (np.average(masked[mask == 0][:, 2]) == 0)

    def test_blue(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = np.ones((3, 3))
        mask = np.pad(mask, 1)
        masked = util.paint_col(test_im, mask, (0, 0, 255))
        assert (np.array_equal(test_im[1:-1, 1:-1], masked[1:-1, 1:-1]))
        assert (np.average(masked[mask == 0][:, 0]) == 0)
        assert (np.average(masked[mask == 0][:, 1]) == 0)
        assert (np.average(masked[mask == 0][:, 2]) == 255)


class TestThresholdBetween(unittest.TestCase):
    def test_no_input(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = util.threshold_between(test_im)
        assert (np.array_equal(mask, np.ones_like(mask)))

    def test_single_channels(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = util.threshold_between(
            test_im,
            x_low=20,
            x_high=200
        )
        test_mask = np.ones_like(mask)
        test_mask[0, 0:2] = 0
        test_mask[4, 1:] = 0
        assert (np.array_equal(mask, test_mask))

        mask = util.threshold_between(
            test_im,
            y_low=20,
            y_high=200
        )
        test_mask = np.ones_like(mask)
        test_mask[0, 0:2] = 0
        test_mask[4, 1:] = 0
        assert (np.array_equal(mask, test_mask))

        mask = util.threshold_between(
            test_im,
            z_low=20,
            z_high=200
        )
        test_mask = np.ones_like(mask)
        test_mask[0, 0:2] = 0
        test_mask[4, 1:] = 0
        assert (np.array_equal(mask, test_mask))

    def test_multi_channels(self):
        test_im = np.dstack(
            [np.arange(0, 250, 10).reshape((5, 5))] * 3
        )
        mask = util.threshold_between(
            test_im,
            x_low=20,
            x_high=200,
            y_low=100,
            y_high=150
        )
        test_mask = np.zeros_like(mask)
        test_mask[2, :] = 1
        test_mask[3, 0] = 1
        assert (np.array_equal(mask, test_mask))


if __name__ == '__main__':
    unittest.main()
