import unittest
import numpy as np
from SpinTools.spinhamiltonian import spinhamiltonian as sh

class TestConstructor(unittest.TestCase):
    def test_types(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, sh.SpinHamiltonian,1)
        self.assertRaises(TypeError, sh.SpinHamiltonian,1j)
        self.assertRaises(TypeError, sh.SpinHamiltonian, True)
        self.assertRaises(TypeError, sh.SpinHamiltonian, None) 