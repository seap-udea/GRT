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

# # Test of GravRay Util

from gravray import *
from gravray.util import *

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# # Test Util Class
# 
# Test suite of the Util submodule of GravRay.

TIMING=1
TEST=1

import unittest

def func(X,factor=1):
    r,q,f=X
    p=factor*1/np.sqrt((2*np.pi)**3)*np.exp(-r**2/2)*r**2*np.cos(q)
    return p

class Test(unittest.TestCase):

    probs=np.array([0.1,0.2,0.40,0.3])
    state=np.array([-2.75666323e+07,1.44279062e+08,3.02263967e+04,
                    -2.97849475e+01,-5.48211971e+00,1.84565202e-05])
    elements=np.array([Const.au,0.5,10.0,30.0,60.0,120.0])
        
    def timing_calc_trig(self):
        Angle.calcTrig(30.0*Angle.Deg)

    def timing_fin2inf(self):
        Util.fin2Inf(0.78,scale=1)

    def timing_inf2fin(self):
        Util.inf2Fin(-5.4,scale=1)

    def timing_gen_index(self):
        Util.genIndex(self.probs)
        
    def timing_trans_state(self):
        Util.transformState(self.state,[Const.km,Const.km/Const.s])
        
    def timing_trans_elements(self):
        Util.transformElements(self.elements,[1/Const.au,Angle.Deg])
        
    def timing_arcdistance_math(self):
        Util.sin=math.sin
        Util.cos=math.cos
        Util.asin=math.asin
        Util.sqrt=math.sqrt
        Util.arcDistance(0,0,90.0*np.pi/180.0,0.0)*180/np.pi        
        
    def timing_arcdistance_numpy(self):
        Util.sin=np.sin
        Util.cos=np.cos
        Util.asin=np.arcsin
        Util.sqrt=math.sqrt
        Util.arcDistance(0,0,90.0*np.pi/180.0,0.0)*180/np.pi

    #"""START COMMENT
    def test_multi_fixed_quad(self):
        nint=MultiQuad(func,["r","q","f"],integrator="fixed_quad")

        
        i,di=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]})
        self.assertEqual(np.isclose([i],0.1209853622595717,rtol=1e-3).tolist(),[True]*1)

        i,di=nint.integrate({"r":[1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]})
        self.assertEqual(np.isclose([i],0.4839414757239004,rtol=1e-3).tolist(),[True]*1)

        i,di=nint.integrate({"r":[0.0,1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]})
        self.assertEqual(np.isclose([i],0.18913556487782582,rtol=1e-3).tolist(),[True]*1)
    
    def test_multi_quad(self):
        nint=MultiQuad(func,["r","q","f"],integrator="quad")

        i,di=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]})
        self.assertEqual(np.isclose([i],0.1209853622595717,rtol=1e-3).tolist(),[True]*1)

        i,di=nint.integrate({"r":[1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]})
        self.assertEqual(np.isclose([i],0.4839414757239004,rtol=1e-3).tolist(),[True]*1)
        
        i,di=nint.integrate({"r":[0.0,1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]})
        self.assertEqual(np.isclose([i],0.19874804309879915,rtol=1e-3).tolist(),[True]*1)
        
    def test_multi_integral(self):
        
        nint=MultiCube(func,["r","q","f"])
        scheme="stroud_cn_5_5"
        nint.setScheme(scheme)
        i=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))
        self.assertEqual(np.isclose([i],0.120985,rtol=1e-3).tolist(),[True]*1)
        
        scheme="stroud_cn_5_5"
        nint.setScheme(scheme)
        i=nint.integrate({"r":[1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]},args=(1.0,))
        self.assertEqual(np.isclose([i],0.483941,rtol=1e-3).tolist(),[True]*1)

        scheme="stroud_cn_5_4"
        nint.setScheme(scheme)
        i=nint.integrate({"r":[0.0,1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]},args=(1.0,))
        self.assertEqual(np.isclose([i],0.198748,rtol=1e-3).tolist(),[True]*1)

    def test_arcdistance(self):
        lat1=Angle.dec((-1,50,3,59))*Angle.Deg
        lat2=Angle.dec((+1,83,38,38))*Angle.Deg
        lon1=Angle.dec((+1,5,42,53))*Angle.Deg
        lon2=Angle.dec((-1,70,4,12))*Angle.Deg
        self.assertAlmostEqual(6371*Util.arcDistance(lon1,lat1,lon2,lat2),15359.006548255658,5)
    
    def test_fin2inf(self):
        try:
            Util.fin2Inf(2)
        except:
            return True
        raise AssertionError("Range not verified for fin2Inf")

    def test_fin2inf(self):
        self.assertAlmostEqual(Util.fin2Inf(0.001,scale=1),-6.906754778648554,13)
        self.assertAlmostEqual(Util.fin2Inf(0.78,scale=1),1.265666373331276,13)
        self.assertAlmostEqual(Util.fin2Inf(np.pi,scale=2*np.pi),0.0,13)

    def test_inf2fin(self):
        self.assertAlmostEqual(Util.inf2Fin(-3.0,scale=1),0.04742587317756678,13)
        self.assertAlmostEqual(Util.inf2Fin(+3.0,scale=1),0.9525741268224334,13)
        self.assertAlmostEqual(Util.inf2Fin(+3.0,scale=10.0),9.525741268224333,13)

    def test_gen_index_error(self):
        try:
            Util.genIndex([0.4,0.4])
        except:
            return True
        raise AssertionError("Error of normalized probabilities not controled")

    def test_gen_index(self):
        N=10000
        M=len(self.probs)
        ns=np.array([Util.genIndex(self.probs) for i in range(N)]+[M])
        h,nx=np.histogram(ns,4)
        self.assertEqual(np.isclose(h/N,
                                    self.probs,
                                    rtol=1e-1).tolist(),
                         [True]*M)

    def test_calc_trig(self):
        self.assertEqual(np.isclose(Angle.calcTrig(30.0*Angle.Deg),
                                    [np.sqrt(3)/2,1./2],
                                    rtol=1e-17).tolist(),
                          [True,True])

    def test_dms(self):
        dms=Angle.dms(293.231241241)
        self.assertEqual(np.isclose(dms,
                                    (1.0,293.0,13.0,52.46846760007429),
                                    rtol=1e-5).tolist(),
                         [True]*4)

    def test_dms_negative(self):
        dms=Angle.dms(-293.231241241)
        self.assertEqual(np.isclose(dms,
                                    (-1.0,293.0,13.0,52.46846760007429),
                                    rtol=1e-5).tolist(),
                         [True]*4)
        
    def test_dec(self):
        dec=Angle.dec((1,5,40,3.4567))
        self.assertAlmostEqual(dec,5.66762686,5)

    def test_dec_negative(self):
        dec=Angle.dec((-1,5,40,3.4567))
        self.assertAlmostEqual(dec,-5.66762686,5)
        
    def test_year(self):
        self.assertEqual(Const.Year,31556736.0)
        
    def test_au(self):
        self.assertEqual(Const.au,1.4959787070000000e11)

    def test_transform_state(self):
        state=Util.transformState(self.state,[Const.km,Const.km/Const.s])
        self.assertEqual(np.isclose(state,
                                    [-2.75666323e+10,1.44279062e+11,3.02263967e+07,
                                     -2.97849475e+04,-5.48211971e+03,1.84565202e-02],
                                    rtol=1e-5).tolist(),
                          [True]*6)
        
    def test_transform_elements(self):
        elements=Util.transformElements(self.elements,[1/Const.au,Angle.Deg])
        self.assertEqual(np.isclose(elements,
                                    [1.,0.5,0.17453293,0.52359878,1.04719755,2.0943951],
                                    rtol=1e-5).tolist(),
                          [True]*6)
    #"""
    #END COMMENT

if __name__=='__main__':
    #Testing
    if TEST:unittest.main(argv=['first-arg-is-ignored'],exit=False)
    
    #Timing Util
    if TIMING:
        print("Timing fin2inf:")
        get_ipython().magic('timeit -n 1000 Test().timing_fin2inf()',scope=globals())

        print("Timing inf2fin:")
        get_ipython().magic('timeit -n 1000 Test().timing_inf2fin()',scope=globals())

        print("Timing genindex:")
        get_ipython().magic('timeit -n 1000 Test().timing_gen_index()',scope=globals())

        print("Trigonometric functions:")
        get_ipython().magic('timeit -n 1000 Test().timing_calc_trig()',scope=globals())

        print("Timing year constant:")
        get_ipython().magic('timeit -n 1000 Const.Year',scope=globals())

        print("Timing elements transformation:")
        get_ipython().magic('timeit -n 1000 Test().timing_trans_elements()',scope=globals())

        print("Timing state transformation:")        
        get_ipython().magic('timeit -n 1000 Test().timing_trans_state()',scope=globals())
        
        print("Timing arcdistance using math functions:")        
        get_ipython().magic('timeit -n 1000 Test().timing_arcdistance_math()',scope=globals())

        print("Timing arcdistance using numpy functions:")        
        get_ipython().magic('timeit -n 1000 Test().timing_arcdistance_numpy()',scope=globals())

