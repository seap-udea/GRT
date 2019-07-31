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
from gravray.spice import *
from gravray.orbit import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# # Test Orbit Class
# 
# Test suite of the Orbit submodule of GravRay.

Spice.loadKernels()

TIMING=1
TEST=1
import unittest
class Test(unittest.TestCase):
    
    #Orbit
    orbit=KeplerianOrbit(1.0)
    
    def test_foo(self):
        pass
    
    #"""
    def test_Jacobian(self):
        self.timing_set_by_elements_ellipse()
        self.orbit.calcJacobians()
        self.assertEqual(np.isclose(self.orbit.Jck.flatten(),
                                    [-0.41654039,-1.23208195,1.07547613,-0.1389311,-0.53730042,-3.08685343,
                                      0.04274803,-5.63319668,-0.62092643,-1.35375627,-1.7177267,-4.09689136,
                                      0.38210857,-1.74958408,1.24185287,0.,-0.39354754,0.62484781,
                                      0.08105459,0.6202556,0.09235913,0.69924506,0.45673547,1.26843361,
                                      0.10757616,-0.91335224,-0.05332357,-0.52685483,-0.43785039,-0.13017475,
                                     -0.01640725,-0.99383322,0.10664714,0.,-0.61446971,-1.1635831],
                                    rtol=1e-5).tolist(),
                          [True]*36)
    def test_jacobians_Ee(self):
        self.timing_set_bound_ellipse()
        self.orbit.calcJacobiansMap()
        self.assertEqual(
            np.isclose(np.diag(self.orbit.JEe),
                       [6.25,4.16666667,1.69765273,1.43239449,2.29183118,3.2228876],
                       rtol=1e-5).tolist(),
            [True]*6
        )

    def test_set_uelements(self):
        self.orbit.setUelements([1.26566637,0.40546511,-1.09861229,-1.60943791,-2.39789527,-2.83321334],0.0)
        self.assertEqual(
            np.isclose(self.orbit.elements,
                       [0.78,0.6,0.78539816,1.04719755,0.52359878,0.34906585],
                       rtol=1e-5).tolist(),
            [True]*6
        )
    
    def test_uelements(self):
        mu=1
        q=0.78
        e=0.6
        i=45.0
        W=60.0
        w=30.0
        M=20.0
        self.orbit.setElements([q,e,i*Angle.Deg,W*Angle.Deg,w*Angle.Deg,M*Angle.Deg],0.0)
        self.orbit.calcUelements()
        self.assertEqual(
            np.isclose(self.orbit.uelements,
                       [1.26566637,0.40546511,-1.09861229,-1.60943791,-2.39789527,-2.83321334],
                       rtol=1e-5).tolist(),
            [True]*6
        )
        
    def test_update_state(self):
        self.timing_set_by_elements_ellipse()
        self.orbit.updateState(10.0)
        self.assertEqual(
            np.isclose(self.orbit.state,
                       [-1.86891304,-4.32426264,-0.54360515,0.1505539,-0.19731752,-0.22904226],
                       rtol=1e-5).tolist(),
            [True]*6
        )
        self.assertAlmostEqual(self.orbit.elements[-1],2.0558356849380304,7)
        self.assertAlmostEqual(self.orbit.celements[-1],2.0558356849380304,7)

    def test_by_elements_ellipse(self):
        self.timing_set_by_elements_ellipse()
        self.orbit.updateState(10.0)
        self.assertEqual(np.isclose(self.orbit.state,
                                    [-1.86891304,-4.32426264,-0.54360515,
                                     0.1505539,-0.19731752,-0.22904226],
                                    rtol=1e-5).tolist(),
                          [True]*6)
        self.assertEqual(np.isclose(self.orbit.derivatives[-2:],
                                    [-0.76518368,0.64381204],
                                    rtol=1e-5).tolist(),
                          [True]*2)
        
    def test_by_elements_hyperbola(self):
        self.timing_set_by_elements_hyperbola()
        self.assertAlmostEqual(self.orbit.celements[0],-2.16666667,7)
        self.assertEqual(np.isclose(self.orbit.state,
                                    [-1.01844517,0.74222555,1.25311217,
                                     -0.97533765,-0.56564603,0.56184417],
                                    rtol=1e-5).tolist(),
                          [True]*6)

    def test_by_elements_ellipse(self):
        self.timing_set_by_elements_ellipse()
        self.assertAlmostEqual(self.orbit.celements[0],3.25,7)
        self.assertEqual(np.isclose(self.orbit.state,
                                    [-1.35375627,0.1389311,1.24185287,
                                     -0.52685483,-0.69924506,0.10664714],
                                    rtol=1e-5).tolist(),
                          [True]*6)

    def test_derivatives(self):
        self.timing_set_by_elements_ellipse()
        self.orbit.calcDerivatives()
        self.assertEqual(np.isclose(self.orbit.derivatives,
                                    [0.17067698,3.25,0.8,2.6,0.72188531,0.69201272],
                                    rtol=1e-5).tolist(),
                          [True]*6)
        
    #"""
    
    def timing_set_by_state_hyperbola(self):
        state=[-1.02,0.74,1.25,-0.98,-0.56,0.56]
        self.orbit.setState(state,0.0)
     
    def timing_set_by_elements_hyperbola(self):
        mu=1
        q=1.3
        e=1.6
        i=45.0
        w=30.0
        W=60.0
        M=20.0
        self.orbit.setElements([q,e,i*Angle.Deg,W*Angle.Deg,w*Angle.Deg,M*Angle.Deg],0.0)

    def timing_set_by_elements_ellipse(self):
        mu=1
        q=1.3
        e=0.6
        i=45.0
        w=30.0
        W=60.0
        M=20.0
        self.orbit.setElements([q,e,i*Angle.Deg,W*Angle.Deg,w*Angle.Deg,M*Angle.Deg],0.0)

    def timing_set_bound_ellipse(self):
        mu=1
        q=0.8
        e=0.6
        i=45.0
        w=30.0
        W=60.0
        M=20.0
        self.orbit.setElements([q,e,i*Angle.Deg,W*Angle.Deg,w*Angle.Deg,M*Angle.Deg],0.0)
        
    #Involved bodies
    earth=Body("EARTH")
    moon=Body("MOON")
    
    #Chelyabinsk impact
    #Time
    tdb_chely=Spice.str2t("02/15/2013 3:20:34 UTC")
    #Location
    chely=Location(earth,61.1*Angle.Deg,54.8*Angle.Deg,23.3*Const.km)
    #Ray
    ray_chely=GrtRay(chely,101.1*Angle.Deg,15.9*Angle.Deg,-18.6*Const.km)
    
    #Moon impact
    tdb_moon=Spice.str2t("2000 JAN 02 12:00:00 UTC")
    #Location
    crater=Location(moon,45.6452*Angle.Deg,41.1274*Angle.Deg,10.0*Const.km)
    #Ray
    ray_crater=GrtRay(crater,1.6502*Angle.Deg,56.981*Angle.Deg,-4.466*Const.km)
    
    #Arbitray impact
    tdb_arb=Spice.str2t("2000 JAN 01 12:00:00 UTC")
    site=Location(earth,25.0*Angle.Deg,53.0*Angle.Deg,100.0*Const.km)
    ray_site=GrtRay(site,40.0*Angle.Deg,20.0*Angle.Deg,-10.0*Const.km)
    
    #"""    
    def test_ray_jacobian_ecliptic(self):
        self.ray_site.updateRay(self.tdb_arb)
        self.ray_site.calcJacobiansBody()
        self.ray_site.calcJacobiansEcliptic()
    
    def test_ray_jacobian(self):
        self.ray_site.updateRay(self.tdb_arb)
        self.ray_site.calcJacobiansBody()

        self.assertEqual(
            np.isclose(self.ray_site.Jcl.flatten(),
                       [-1.65111080e+06,-4.68755962e+06,5.45429642e-01,0.00000000e+00,0.00000000e+0,0.00000000e+00,
                        3.54081854e+06,-2.18584495e+06,2.54338019e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,
                        0.00000000e+00,3.89749442e+06,7.98635510e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,
                        3.65693764e+03,6.56078885e+03,-1.84959827e-05,-1.32977275e+03,-7.95087289e+03,-5.89786728e-01,
                        5.77779511e+03,2.64433327e+03,3.96647629e-05,-8.56270845e+03,-1.28181727e+03,3.91443323e-01,
                        0.00000000e+00,3.69061965e+03,0.00000000e+00,3.63509979e+03,-5.92794777e+03,7.06345341e-01],
                       rtol=1e-5).tolist(),
                       [True]*36)

    def test_propagate_moon(self):
        self.timing_propagate_ray_moon()
        E=np.copy(self.ray_crater.terminal.elements)
        Const.transformElements(E,[1/Const.au,Angle.Rad],implicit=True)
        self.assertEqual(np.isclose(E,
                                    [9.28310700e-01,4.62115646e-02,6.82524233e+00,
                                     2.82253534e+02,2.81089088e+02,2.59457851e+02],
                                    rtol=1e-5).tolist(),
                         [True]*6)        

    def test_propagate_chely(self):
        self.timing_propagate_ray_earth()
        E=np.copy(self.ray_chely.terminal.elements)
        Const.transformElements(E,[1/Const.au,Angle.Rad],implicit=True)
        self.assertEqual(np.isclose(E,
                                    [0.73858102,0.54966922,4.04158232,
                                     326.57255475,106.86339209,21.32411541],
                                    rtol=1e-5).tolist(),
                         [True]*6)

    def test_state_chely(self):
        self.ray_chely.updateRay(self.tdb_chely)

        self.assertEqual(np.isclose(self.ray_chely.stateBody,
                                    [1.78729411e+06,3.23767769e+06,5.20762055e+06,
                                     1.23526608e+04,-1.33886204e+04,-2.17876404e+03],
                                    rtol=1e-5).tolist(),
                         [True]*6)
        self.assertEqual(np.isclose(self.ray_chely.stateEcl,
                                    [-8.82111395e+05,-1.22185261e+06,6.20687075e+06, 
                                     -1.53985024e+04,8.07537430e+03,-5.85361851e+03],
                                    rtol=1e-5).tolist(),
                         [True]*6)

    #"""
    def timing_update_ray(self):
        self.ray_chely.updateRay(self.tdb_chely)

    def timing_propagate_ray_earth(self):
        self.ray_chely.updateRay(self.tdb_chely)
        self.ray_chely.propagateRay()

    def timing_propagate_ray_moon(self):
        self.ray_crater.updateRay(self.tdb_moon)
        self.ray_crater.propagateRay()
    
if __name__=='__main__':
    #Testing
    unittest.main(argv=['first-arg-is-ignored'],exit=False)

    #"""
    print("Timing set by elements hyperbola:")
    get_ipython().magic('timeit -n 100 Test().timing_set_by_elements_hyperbola()')

    print("Timing set by elements ellipse:")
    get_ipython().magic('timeit -n 100 Test().timing_set_by_elements_ellipse()')

    print("Timing set by state hyperbola:")
    get_ipython().magic('timeit -n 100 Test().timing_set_by_state_hyperbola()')

    print("Timing Jacobian Jck:")
    t=Test()
    get_ipython().magic('timeit -n 100 t.orbit.calcJacobians()')

    print("Timing Determinant of Jacobian:")
    get_ipython().magic('timeit -n 100 np.linalg.det(t.orbit.Jck)')

    print("Timing update state:")
    t=Test()
    get_ipython().magic('timeit -n 1000 t.orbit.updateState(10.0)')

    print("Timing Jacobians Ee:")
    t=Test()
    t.timing_set_bound_ellipse()
    get_ipython().magic('timeit -n 100 t.orbit.calcJacobiansMap()')

    #Timing
    print("Timing update ray:")
    get_ipython().magic('timeit -n 100 Test().timing_update_ray()')

    print("Timing propagate ray earth:")
    get_ipython().magic('timeit -n 100 Test().timing_propagate_ray_earth()')

    print("Timing propagate ray moon:")
    get_ipython().magic('timeit -n 100 Test().timing_propagate_ray_moon()')

