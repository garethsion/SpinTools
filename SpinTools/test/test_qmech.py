import unittest
import numpy as np
from SpinTools.qmech import qmech

class TestAngularMomentum(unittest.TestCase):
    def test_area(self):
        # Test areas when radius >= 0
        self.assertTrue(np.array(qmech.Qmech().angular_momentum(0.5)['x']==np.matrix('0,.5;.5,0')).all())
        self.assertTrue(np.array(qmech.Qmech().angular_momentum(0.5)['y']==1j*np.matrix('0.,.5;0-.5,0')).all())
        self.assertTrue(np.array(qmech.Qmech().angular_momentum(0.5)['z']==np.matrix('.5,0;0,-0.5')).all())

    def test_values(self):
        # Make sure errors are raised when necessary
        self.assertRaises(ValueError, qmech.Qmech().angular_momentum, -1)

    def test_types(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, qmech.Qmech().angular_momentum, 1j) 
        self.assertRaises(TypeError, qmech.Qmech().angular_momentum, "J")
        self.assertRaises(TypeError, qmech.Qmech().angular_momentum, True)
        self.assertRaises(TypeError, qmech.Qmech().angular_momentum, None)

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