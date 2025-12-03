import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Scripts')))
from Scripts.calc import add, subtract, multiply, divide

class TestCalc(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)


    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)


    def test_divide(self):
        self.assertEqual(divide(6, 3), 2)

        with self.assertRaises(ValueError):
            divide(5, 0)    