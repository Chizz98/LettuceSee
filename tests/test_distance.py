import lettuceSee as lS
import unittest
import numpy as np
import scipy.ndimage as ndi


class TestImDistMap(unittest.TestCase):
    def test_one_object_one_seed(self):
        mask = np.ones((5, 5))
        mask = np.pad(mask, pad_width=1)
        seeds = np.zeros_like(mask)
        seeds[3, 3] = 1
        seeds = seeds.astype(bool)
        distance = lS.distance.im_dist_map(mask, seeds.astype(bool))
        # Set up test case
        test_dist = ndi.distance_transform_edt(np.invert(seeds))
        test_dist[mask == 0] = 0
        assert(np.array_equal(test_dist, distance))

    def test_multiple_objects(self):
        mask = np.ones((5, 5))
        mask = np.pad(mask, pad_width=1)
        mask = np.concatenate((mask, mask), axis=1)
        seeds = np.zeros_like(mask)
        seeds[3, 3] = 1
        seeds = seeds.astype(bool)
        distance = lS.distance.im_dist_map(mask, seeds.astype(bool))

        # Set up test case
        test_dist = ndi.distance_transform_edt(np.invert(seeds))
        test_dist[mask == 0] = 0
        assert (np.array_equal(test_dist, distance))

    def test_multiple_seeds(self):
        mask = np.ones((5, 5))
        mask = np.pad(mask, pad_width=1)
        mask = np.concatenate((mask, mask), axis=1)
        seeds = np.zeros_like(mask)
        seeds[1, 1] = 1
        seeds[1, 5] = 1
        seeds = seeds.astype(bool)
        distance = lS.distance.im_dist_map(mask, seeds.astype(bool))

        # Set up test case
        test_dist = ndi.distance_transform_edt(np.invert(seeds))
        test_dist[mask == 0] = 0
        assert(np.array_equal(test_dist, distance))


if __name__ == '__main__':
    unittest.main()
