import unittest
import numpy as np
from spectral_segmentation import kmeans, spectral_cluster

class TestClustering(unittest.TestCase):
    def test_kmeans(self):
        data = np.array([[0, 0], [0, 1], [5, 5], [5, 6]], dtype=float)
        labels = kmeans(data, 2, max_iter=50)
        self.assertEqual(len(labels), 4)
        self.assertEqual(set(labels), {0, 1})

    def test_spectral_cluster(self):
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=float)
        labels = spectral_cluster(adj, 2)
        self.assertEqual(len(labels), 4)
        self.assertEqual(set(labels), {0, 1})

if __name__ == '__main__':
    unittest.main()
