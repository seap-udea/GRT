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

from gravray import *
from gravray.util import *
from gravray.stat import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# # Test Statistics Class
# 
# Test suite of the Statistics submodule of GravRay.

#Unitary test
class Test(unittest.TestCase):

    mnd=MultiNormal()
    
    #"""
    def test_rvs(self):
        self.timing_set_unflatten()
        r=self.mnd.rvs(1)
        print(r)
    
    def test_pdf(self):
        self.timing_set_unflatten()
        p=self.mnd.pdf(self.mnd.locs[0])
        self.assertAlmostEqual(p,0.09989816167966402,7)
        
    
    def test_set_unflatten(self):
        self.timing_set_unflatten()
        self.assertEqual(self.mnd.M,2)
        self.assertEqual(self.mnd.N,10*self.mnd.M-1)
        
        self.assertEqual(np.isclose(self.mnd.aweights,
                                    [0.6,0.4],
                                    rtol=1e-5).tolist(),
                         [True]*2)
        self.assertEqual(np.isclose(self.mnd.params,
                                    [0.6, 
                                     0.5, 0.5, -2.0, 
                                     2.0, 0.3, -2.6, 
                                     1.3, 0.7, 0.5, 
                                     0.4, 0.9, 1.6, 
                                     -0.6981317007977318, -1.5009831567151235, 0.0, 
                                     1.3962634015954636, -1.9024088846738192, 0.0],
                                    rtol=1e-5).tolist(),
                         [True]*self.mnd.N)

        self.assertEqual(np.isclose(self.mnd.covs[0].flatten(),
                                    [1.09550921,-0.70848654,-0.01073505,
                                     -0.70848654,0.84565862,-0.01279353,
                                     -0.01073505,-0.01279353,0.48883217],
                                    rtol=1e-5).tolist(),
                         [True]*9)

        self.assertEqual(np.isclose(self.mnd.covs[1].flatten(),
                                    [2.30773378,-0.37870341,0.53051967,
                                     -0.37870341,0.22677563,-0.09354493,
                                     0.53051967,-0.09354493,0.99549059],
                                    rtol=1e-5).tolist(),
                         [True]*9)

    def test_set_flatten(self):
        self.timing_set_flatten()
        self.assertEqual(self.mnd.M,2)
        self.assertEqual(self.mnd.N,10*self.mnd.M-1)
        self.assertEqual(np.isclose(self.mnd.params,
                                    [0.6, 
                                     0.5, 0.5, -2.0, 
                                     2.0, 0.3, -2.6, 
                                     1.3, 0.7, 0.5, 
                                     0.4, 0.9, 1.6, 
                                     -0.6981317007977318, -1.5009831567151235, 0.0, 
                                     1.3962634015954636, -1.9024088846738192, 0.0],
                                    rtol=1e-5).tolist(),
                         [True]*self.mnd.N)

        self.assertEqual(np.isclose(self.mnd.covs[0].flatten(),
                                    [1.09550921,-0.70848654,-0.01073505,
                                     -0.70848654,0.84565862,-0.01279353,
                                     -0.01073505,-0.01279353,0.48883217],
                                    rtol=1e-5).tolist(),
                         [True]*9)

        self.assertEqual(np.isclose(self.mnd.covs[1].flatten(),
                                    [2.30773378,-0.37870341,0.53051967,
                                     -0.37870341,0.22677563,-0.09354493,
                                     0.53051967,-0.09354493,0.99549059],
                                    rtol=1e-5).tolist(),
                         [True]*9)

    #""" 
    
    def timing_set_unflatten(self):
        weights=[0.6]
        locs=[
            [0.5,0.5,-2.0],
            [2.0,0.3,-2.6]
        ]
        scales=[
            [1.3,0.7,0.5],
            [0.4,0.9,1.6]
        ]
        angles=[
            [-40.0*Angle.Deg,-86.0*Angle.Deg,0.0*Angle.Deg],
            [+80.0*Angle.Deg,-109.0*Angle.Deg,0.0*Angle.Deg]
        ]
        self.mnd.setUnflatten(weights,locs,scales,angles)
        
    def timing_set_flatten(self):
        params=[
            #weights
            0.6,
            #locs
            0.5, 0.5, -2.0,
            2.0, 0.3, -2.6, 
            #scales
            1.3, 0.7, 0.5,
            0.4, 0.9, 1.6, 
            #Angles
            -40.0*Angle.Deg,-86.0*Angle.Deg,0.0*Angle.Deg,
            +80.0*Angle.Deg,-109.0*Angle.Deg,0.0*Angle.Deg
        ]
        self.mnd.setFlatten(params)

if __name__=='__main__':
    #Testing
    unittest.main(argv=['first-arg-is-ignored'],exit=False)
    
    #Timing
    #"""
    print("Timing set unflatten:")
    get_ipython().magic('timeit -n 100 Test().timing_set_unflatten()')

    print("Timing set flatten:")
    get_ipython().magic('timeit -n 100 Test().timing_set_flatten()')
    
    print("Timing PDF:")
    t=Test()
    get_ipython().magic('timeit -n 100 t.mnd.pdf([0,0,0])')

    print("Timing RVS:")
    t=Test()
    get_ipython().magic('timeit -n 100 t.mnd.rvs(1)')
    #"""

