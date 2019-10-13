import unittest
import numpy as np
import qmech

class TestEnlargeMatrix(unittest.TestCase):
    def test_expand(self):
        # Test size of expanded matrices is correct
        sx = qmech.Qmech().angular_momentum(.5)['x']
        self.assertTrue(np.shape(qmech.Qmech().enlarge_matrix(1,sx)),(2,2))
        self.assertTrue(np.shape(qmech.Qmech().enlarge_matrix(2,sx)),(4,4))
        self.assertTrue(np.shape(qmech.Qmech().enlarge_matrix(3,sx)),(6,6))

    # def test_values(self):
    #     # Make sure errors are raised when necessary
    #     self.assertRaises(ValueError, qmech.Qmech().angular_momentum, -1)

    def test_types(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, qmech.Qmech().enlarge_matrix, 'm','j') 
        self.assertRaises(TypeError, qmech.Qmech().enlarge_matrix, True, True)
        self.assertRaises(TypeError, qmech.Qmech().enlarge_matrix, None, None)

