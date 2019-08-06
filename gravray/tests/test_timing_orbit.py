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

# # Test of GravRay Orbit

from gravray import *
from gravray.util import *
from gravray.spice import *
from gravray.orbit import *

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

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
    ray_site=GrtRay(site,40.0*Angle.Deg,20.0*Angle.Deg,-20.0*Const.km)

    def timing_update_ray(self):
        self.ray_chely.updateRay(self.tdb_chely)

    def timing_propagate_ray_earth(self):
        self.ray_chely.propagateRay(self.tdb_chely)

    def timing_propagate_ray_moon(self):
        self.ray_crater.propagateRay(self.tdb_moon)

    def timing_set_by_state_hyperbola(self):
        state=[-1.02,0.74,1.25,-0.98,-0.56,0.56]
        self.orbit.setState(state,0.0)
     
    def timing_set_by_elements_ellipse(self):
        mu=1
        q=1.3
        e=0.6
        i=45.0
        w=30.0
        W=60.0
        M=20.0
        self.orbit.setElements([q,e,i*Angle.Deg,W*Angle.Deg,w*Angle.Deg,M*Angle.Deg],0.0)

    def timing_set_by_elements_hyperbola(self):
        mu=1
        q=1.3
        e=1.6
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

    def test_collision(self):
        t=414170434.0 
        lon=204.014122*Angle.Deg
        lat=-30.0*Angle.Deg
        H=23300.0
        A=227.241656*Angle.Deg
        h=48.590378*Angle.Deg
        v=11200.0
        site=Location(self.earth,lon,lat,H)
        ray=GrtRay(site,A,h,v)
        ray.updateRay(t)
        try:
            ray.propagateRay(t)
        except AssertionError as e:
            return True
        raise AssertionError("No error control")
        
    """ START COMMENT
    def test_update_by_time(self):
        self.ray_chely.propagateRay(self.tdb_chely)
        t=self.ray_chely.states[0][0]
        mu=self.ray_chely.states[0][1]
        state=deepcopy(self.ray_chely.states[0][3])
        orbit=KeplerianOrbit(mu)
        orbit.setState(state,t)
        Md,deltat,state=orbit.calcStateByTime(414068235.2925538)
        self.assertEqual(np.isclose([Md,deltat],
                                    [-790.1856733717548,-102198.707446203],rtol=1e-5).tolist(),[True]*2)
        self.assertEqual(np.isclose(state,
                                    [ 1.33667947e+09,-6.32323959e+08,2.41440625e+08,
                                     -1.30172709e+04,6.13849049e+03,-2.27590686e+03],
                                    rtol=1e-5).tolist(),[True]*6)
        
    def test_update_by_distance(self):
        self.ray_chely.propagateRay(self.tdb_chely)
        t=self.ray_chely.states[0][0]
        mu=self.ray_chely.states[0][1]
        state=deepcopy(self.ray_chely.states[0][3])
        orbit=KeplerianOrbit(mu)
        orbit.setState(state,t)
        Md,deltat,state=orbit.calcStateByDistance(self.earth.rhill,direction=-1)        
        self.assertEqual(np.isclose([Md,deltat],
                                    [-790.1856733717548,-102198.707446203],rtol=1e-5).tolist(),[True]*2)
        self.assertEqual(np.isclose(state,
                                    [ 1.33667947e+09,-6.32323959e+08,2.41440625e+08,
                                     -1.30172709e+04,6.13849049e+03,-2.27590686e+03],
                                    rtol=1e-5).tolist(),[True]*6)
        
    def test_ray_detjacobian_chely(self):
        self.ray_chely.propagateRay(self.tdb_chely)
        detJ=self.ray_chely.calcJacobianDeterminant()
        self.assertEqual(np.isclose([detJ],[-5.369756753915885e-13],rtol=1e-5).tolist(),[True])

        #Numerical Jacobian
        def Rimp2Ehel(X):
            global tdb
            lon,lat,alt,A,h,v=X
            site=Location(self.earth,lon,lat,alt)
            ray=GrtRay(site,A,h,v)
            ray.propagateRay(self.tdb_chely)
            hel=ray.terminal
            return hel.celements
        X=self.ray_chely.Rimp
        dX=np.abs(X*1e-5)
        y,Jhi=Jacobians.computeNumericalJacobian(Rimp2Ehel,X,dX)
        detJnum=np.linalg.det(Jhi)

    def test_by_elements_ellipse(self):
        self.timing_set_by_elements_ellipse()
        self.assertAlmostEqual(self.orbit.celements[0],3.25,7)
        self.assertEqual(np.isclose(self.orbit.state,
                                    [-1.35375627,0.1389311,1.24185287,
                                     -0.52685483,-0.69924506,0.10664714],
                                    rtol=1e-5).tolist(),
                          [True]*6)
        self.assertEqual(np.isclose(self.orbit.secondary,
                                    [0.17067698,3.25,0.8,2.6,0.72188531,0.69201272],
                                    rtol=1e-5).tolist(),
                          [True]*6)
        
    def test_by_elements_hyperbola(self):
        self.timing_set_by_elements_hyperbola()
        self.assertAlmostEqual(self.orbit.celements[0],-2.16666667,7)
        self.assertEqual(np.isclose(self.orbit.state,
                                    [-1.01844517,0.74222555,1.25311217,
                                     -0.97533765,-0.56564603,0.56184417],
                                    rtol=1e-5).tolist(),
                         [True]*6)

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
        M,dt,state=self.orbit.calcStateByTime(10.0)
        self.assertEqual(
            np.isclose(state,
                       [-1.86891304,-4.32426264,-0.54360515,0.1505539,-0.19731752,-0.22904226],
                       rtol=1e-5).tolist(),
            [True]*6
        )
    
    def test_keplerian_jacobian(self):
        self.timing_set_by_elements_ellipse()
        Jck=Jacobians.calcKeplerianJacobians(self.orbit.mu,self.orbit.celements,self.orbit.state)
        self.assertEqual(np.isclose(Jck.flatten(),
                                    [-0.41654039,-1.23208195,1.07547613,-0.1389311,-0.53730042,-3.08685343,
                                      0.04274803,-5.63319668,-0.62092643,-1.35375627,-1.7177267,-4.09689136,
                                      0.38210857,-1.74958408,1.24185287,0.,-0.39354754,0.62484781,
                                      0.08105459,0.6202556,0.09235913,0.69924506,0.45673547,1.26843361,
                                      0.10757616,-0.91335224,-0.05332357,-0.52685483,-0.43785039,-0.13017475,
                                     -0.01640725,-0.99383322,0.10664714,0.,-0.61446971,-1.1635831],
                                    rtol=1e-5).tolist(),
                          [True]*36)

    def test_map_jacobian(self):
        mu=1
        q=0.78
        e=0.6
        i=45.0
        W=60.0
        w=30.0
        M=20.0
        self.orbit.setElements([q,e,i*Angle.Deg,W*Angle.Deg,w*Angle.Deg,M*Angle.Deg],0.0)
        self.orbit.calcUelements()
        JEe,JeE=Jacobians.calcMapJacobian(self.orbit.elements,[1,1,np.pi,2*np.pi,2*np.pi,2*np.pi])
        self.assertEqual(np.isclose(np.diag(JEe),
                                    [5.82750583,4.16666667,1.69765273,1.14591559,2.08348289,3.03330597],
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

    def test_propagate_chely(self):
        self.timing_propagate_ray_earth()
        E=np.copy(self.ray_chely.terminal.elements)
        Util.transformElements(E,[1/Const.au,Angle.Rad],implicit=True)
        self.assertEqual(np.isclose(E,
                                    [0.73858152,0.54966804,4.0415789,
                                     326.57255584,106.8634198,21.32354715],
                                    rtol=1e-5).tolist(),
                         [True]*6)

    def test_propagate_moon(self):
        self.timing_propagate_ray_moon()
        E=np.copy(self.ray_crater.terminal.elements)
        Util.transformElements(E,[1/Const.au,Angle.Rad],implicit=True)
        self.assertEqual(np.isclose(E,
                                    [9.28278830e-01,4.62067544e-02,6.82258090e+00,
                                     2.82254946e+02,2.81138672e+02,2.59270699e+02],
                                    rtol=1e-5).tolist(),
                         [True]*6)        

    def test_ray_jacobian(self):
        self.ray_chely.updateRay(self.tdb_chely)
        Jcl,Jel=Jacobians.calcImpactJacobian(self.ray_chely.body,
                                             self.ray_chely.Rimp,
                                             self.ray_chely.stateBody)        
        self.assertEqual(
            np.isclose(Jel.flatten(),
                       [+3.58981963e+06,1.26210772e+06,-1.37484659e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,
                        -8.17426203e+05,6.12568420e+06,-1.88278200e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,
                        +3.49265604e+05,1.36445068e+06,9.72445005e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00,
                        -9.73737363e+03,-9.00420338e+02,4.06904053e-05,6.80383044e+03,7.11968399e+03,8.41911908e-01,
                        -1.41203876e+04,-4.14362932e+03,-9.26548041e-06,1.60363241e+04,1.32400724e+03,-4.37355872e-01,
                        +6.13527450e+03,-4.47105670e+03,3.95890614e-06,4.06677244e+03,-1.71323409e+04,3.16076227e-01],
                       rtol=1e-5).tolist(),
                       [True]*36)
    #"""
    #END COMMENT

from gravray import *
from gravray.util import *
from gravray.spice import *
from gravray.orbit import *
if __name__=='__main__':
    #Testing
    if TEST:unittest.main(argv=['first-arg-is-ignored'],exit=False)

    if TIMING:
        print("Timing set by elements hyperbola:")
        get_ipython().magic('timeit -n 100 Test().timing_set_by_elements_hyperbola()',scope=globals())

        print("Timing set by elements ellipse:")
        get_ipython().magic('timeit -n 100 Test().timing_set_by_elements_ellipse()',scope=globals())

        print("Timing set by state hyperbola:")
        get_ipython().magic('timeit -n 100 Test().timing_set_by_state_hyperbola()',scope=globals())

        print("Timing update state:")
        t=Test()
        get_ipython().magic('timeit -n 1000 t.orbit.calcStateByTime(10.0)',scope=globals())

        #Timing
        print("Timing update ray:")
        get_ipython().magic('timeit -n 100 Test().timing_update_ray()',scope=globals())

        print("Timing propagate ray earth:")
        get_ipython().magic('timeit -n 100 Test().timing_propagate_ray_earth()',scope=globals())

        print("Timing propagate ray moon:")
        get_ipython().magic('timeit -n 100 Test().timing_propagate_ray_moon()',scope=globals())

