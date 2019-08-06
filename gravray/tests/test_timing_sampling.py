#################################################################################
#     ______                 ____             ___                               # 
#    / ____/________ __   __/ __ \____ ___  _|__ \                              #
#   / / __/ ___/ __ `/ | / / /_/ / __ `/ / / /_/ /                              #
#  / /_/ / /  / /_/ /| |/ / _, _/ /_/ / /_/ / __/                               #
#  \____/_/   \__,_/ |___/_/ |_|\__,_/\__, /____/                               #
#                                    /____/                                     #
#################################################################################
# Jorge I. Zuluaga (C) 2019                                                     #
#################################################################################
#!/usr/bin/env python
# coding: utf-8

# # Test of GravRay Sampling

from gravray import *
from gravray.sampling import *

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib nbagg')

# # Test Sampling Class
# 
# Test suite of the Sampling submodule of GravRay.

TIMING=1
TEST=1

#Unitary test
import unittest
class Test(unittest.TestCase):

    s2d=Sample(1000)
    s3d=Sample(1000)
    
    #"""
    def test_dimensions(self):
        self.timing_unit_circle()
        self.assertEqual(self.s2d.dim,2)
        self.timing_unit_hemisphere()
        self.assertEqual(self.s3d.dim,3)

    def test_distance(self):
        self.timing_unit_circle()
        self.s2d._calcDistances()
        self.assertAlmostEqual(self.s2d.dstar,1.7,1)
        self.timing_unit_sphere()
        self.s3d._calcDistances()
        self.assertAlmostEqual(self.s3d.dstar,3.4,1)
        self.timing_unit_hemisphere()
        self.s3d._calcDistances()
        self.assertAlmostEqual(self.s3d.dstar,2.4,1)
        
    def test_polar(self):
        self.timing_unit_hemisphere()
        suma=0
        for i,p in enumerate(self.s3d.pp):
            r=np.array([np.cos(p[2])*np.cos(p[1]),np.cos(p[2])*np.sin(p[1]),np.sin(p[2])])
            suma+=np.linalg.norm(r-self.s3d.ss[i])
        self.assertAlmostEqual(suma,0,5)   
    #""" 
    
    def timing_unit_circle(self):
        np.random.seed(10)
        self.s2d.genUnitCircle()

    def timing_unit_hemisphere(self):
        np.random.seed(10)
        self.s3d.genUnitHemisphere()

    def timing_unit_cosine(self):
        np.random.seed(10)
        self.s3d.genCosineWeightedUnitHemisphere()

    def timing_unit_sphere(self):
        np.random.seed(10)
        self.s3d.genUnitSphere()

if __name__=='__main__':
    #Testing
    if TEST:unittest.main(argv=['first-arg-is-ignored'],exit=False)
    
    if TIMING:
        #Timing
        print("Timing unit circle:")
        get_ipython().magic('timeit -n 10 Test().timing_unit_circle()',scope=globals())

        print("Timing unit hemisphere:")
        get_ipython().magic('timeit -n 10 Test().timing_unit_hemisphere()',scope=globals())

        print("Timing unit hemisphere (cosine weighted):")
        get_ipython().magic('timeit -n 10 Test().timing_unit_cosine()',scope=globals())

        print("Timing unit sphere:")
        get_ipython().magic('timeit -n 10 Test().timing_unit_sphere()',scope=globals())

