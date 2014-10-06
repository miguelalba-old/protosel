import unittest
import numpy as np
import protosel


def load_orings():

    # Load Challenger USA Space Shuttle O-Ring data set
    # Attributes information:
    # 1st attribute: Number of o-rings at risk on a given flight
    # 2nd attribute: Number experiencing thermal distress
    # 3rd attribute: Launch temperature (degrees F)
    # 4th attribute: Leak-check pressure (psi)
    # 5th attribute: Temporal order of flight
    dataset = np.array([
        [6, 0, 66,  50,  1],
        [6, 1, 70,  50,  2],
        [6, 0, 69,  50,  3],
        [6, 0, 68,  50,  4],
        [6, 0, 67,  50,  5],
        [6, 0, 72,  50,  6],
        [6, 0, 73, 100,  7],
        [6, 0, 70, 100,  8],
        [6, 1, 57, 200,  9],
        [6, 1, 63, 200, 10],
        [6, 1, 70, 200, 11],
        [6, 0, 78, 200, 12],
        [6, 0, 67, 200, 13],
        [6, 2, 53, 200, 14],
        [6, 0, 67, 200, 15],
        [6, 0, 75, 200, 16],
        [6, 0, 70, 200, 17],
        [6, 0, 81, 200, 18],
        [6, 0, 76, 200, 19],
        [6, 0, 79, 200, 20],
        [6, 0, 75, 200, 21],
        [6, 0, 76, 200, 22],
        [6, 1, 58, 200, 23]
    ])

    data = dataset[:, np.arange(5) != 1]
    target = dataset[:, 1]
    return data, target


class TestISA(unittest.TestCase):
    """Test an instance selection algorithm."""

    @classmethod
    def setUpClass(self):
        # Load test dataset
        self.data, self.target = load_orings()

    def test_enn(self):
        obtained_results = protosel.enn(self.data, self.target).tolist()
        expected_results = [True, False, True, True, True, True, True, True,
                            True, True, False, True, True, False, True, True,
                            True, True, True, True, True, True, False]
        self.assertEqual(obtained_results, expected_results)

    def test_renn(self):
        obtained_results = protosel.renn(self.data, self.target).tolist()
        expected_results = [True, False, True, True, True, True, True, True,
                            True, True, False, True, True, False, True, True,
                            True, True, True, True, True, True, False]
        self.assertEqual(obtained_results, expected_results)

    def test_allknn(self):
        obtained_results = protosel.allknn(self.data, self.target).tolist()
        expected_results = [True, False, True, True, True, True, True, True,
                            True, False, False, True, True, False, True, True,
                            True, True, True, True, True, True, False]
        self.assertEqual(obtained_results, expected_results)


if __name__ == '__main__':
    unittest.main()
