from module import *
import unittest

class Test(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum(1,1),2)

    def test_type_s1(self):
        try:
            sum("a",2)
        except TypeError:
            assert True
        except:
            self.assertTrue(False,msg="No type error raised")

    def test_type_s2(self):
        try:
            sum(1,"b")
        except TypeError:
            assert True
        except:
            self.assertTrue(False,msg="No type error raised")

