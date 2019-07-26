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

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# # Test Spice Class
# 
# Test suite of the Util submodule of GravRay.

Spice.loadKernels()

class Test(unittest.TestCase):

    #"""
    def test_load_kernels(self):
        Spice.loadKernels()

    def test_shape(self):
        Spice.calcShape("EARTH")
        self.assertEqual(np.isclose([Spice.Ra["EARTH"],Spice.f["EARTH"]],
                                    [6378.1366e3,0.0033528131084554717],
                                    rtol=1e-5).tolist(),
                          [True,True])
        
    def test_string_tdb(self):
        self.assertAlmostEqual(Spice.str2t("2000 JAN 01 12:00:00"),0.0,7)
        
    def test_right_kernels(self):
        pass
    
    def timing_load_kernels(self):
        Spice.loadKernels()
    #"""
    
    earth=Body("EARTH")
    moon=Body("MOON")

    #"""
    def test_body_rhill(self):
        self.timing_update_earth()
        self.assertEqual(np.isclose([self.earth.rhill],[1496558526],rtol=1e-5).tolist(),[True])

    def test_moon_rhill(self):
        self.timing_update_moon()        
        self.assertEqual(np.isclose([self.moon.rhill],[61460054],rtol=1e-5).tolist(),[True])

    def test_body_shape(self):
        Spice.calcShape("EARTH")
        self.assertEqual(np.isclose([self.earth.Ra,self.earth.f],
                                    [6378.1366e3,0.0033528131084554717],
                                    rtol=1e-5).tolist(),
                          [True,True])
            
    def test_body_state(self):
        self.timing_update_earth()
        self.assertEqual(np.isclose(self.earth.stateHelio,
                                    [-2.75666323e+10,1.44279062e+11,3.02263967e+07,
                                     -2.97849475e+04,-5.48211971e+03,1.84565202e-02],
                                    rtol=1e-5).tolist(),
                          [True]*6)

    def test_body_transform(self):
        self.timing_update_earth()
        self.assertEqual(np.isclose(self.earth.Tbod2ecl.flatten(),
                                    [ 1.76980593e-01,9.84214341e-01,-2.51869708e-05,
                                     -9.03007988e-01,1.62388314e-01,3.97751944e-01,
                                     3.91477257e-01,-7.03716309e-02,9.17492992e-01],
                                    rtol=1e-5).tolist(),
                          [True]*9)
    
    def test_moon_state(self):
        self.timing_update_moon()
        self.assertEqual(np.isclose(self.moon.stateHelio,
                                    [-2.78582406e+10,1.44004083e+11,6.64975943e+07,
                                     -2.91414161e+04,-6.21310369e+03,-1.14880075e+01],
                                    rtol=1e-5).tolist(),
                          [True]*6)
        self.assertEqual(np.isclose(self.moon.Tbod2ecl.flatten(),
                                    [0.78422705,-0.62006192,-0.02260867,
                                     0.61987147,0.78455064,-0.01548052,
                                     0.02733653,-0.00187423,0.99962453],
                                    rtol=1e-5).tolist(),
                          [True]*9)
        
    def timing_update_earth(self):
        Spice.loadKernels()
        tdb=Spice.str2t("2000 JAN 01 12:00:00")
        self.earth.updateBody(tdb)

    def timing_update_moon(self):
        Spice.loadKernels()
        tdb=Spice.str2t("2000 JAN 01 12:00:00")
        self.moon.updateBody(tdb)
        
    #Objects
    moon=Body("MOON")
    earth=Body("EARTH")

    #Moon impact
    tdb_moon=spy.str2et("2000 JAN 02 11:58:56 UTC")
    crater=Location(moon,45.6452*Angle.Deg,41.1274*Angle.Deg,10.0*Const.km)

    #Earth impact
    tdb_earth=spy.str2et("2000 JAN 01 12:00:00 UTC")
    impact=Location(earth,0*Angle.Deg,0*Angle.Deg,0*Const.km)
    
    #Chelyabinsk impact
    tdb_chely=spy.str2et("02/15/2013 3:20:34 UTC")
    chely=Location(earth,61.1*Angle.Deg,54.8*Angle.Deg,23.3*Const.km)
    
    def test_vbod2loc(self):
        #These are the components of the Chelyabinsk-impactor velocity as reported by CNEOS
        vBod=np.array([12.8,-13.3,-2.4])
        A,h,vimp=self.chely.vbod2loc(-vBod)
        self.assertAlmostEqual(A*Angle.Rad,99.8961127649985,5)
        self.assertAlmostEqual(h*Angle.Rad,15.92414245029081,5)
        
    def test_loc2vbod(self):
        #These is the radiant and speed (18.6 km/s) of the chelyabinsk impact 
        vBod=self.chely.loc2vbod(101.1*Angle.Deg,+15.9*Angle.Deg,-18.6)
        self.assertEqual(np.isclose(vBod,
                                    np.array([12.8,-13.1,-2.4]),
                                    rtol=1e-1).tolist(),
                         [True]*3)
        
    def test_earth_impact(self):
        self.impact.updateLocation(self.tdb_earth)

        #Body and location properties
        self.assertEqual(np.isclose(self.impact.Tloc2bod.flatten(),
                                    [0.,0.,1.,0.,1.,0.,1.,0.,0.],
                                    rtol=1e-5).tolist(),
                          [True]*9)        
        self.assertEqual(np.isclose(self.impact.posLocal,
                                    [0.,0.,0.],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.impact.velLocal,
                                    [0.,463.83118255,0.],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.impact.posBody,
                                    [6378136.6,0.,0.],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.impact.velBody,
                                    [0.,463.83118255,0.],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.impact.posEcl,
                                    [1158174.70642354,-5754597.59496353,2494767.39551009],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.impact.velEcl,
                                    [456.12009587,77.28027145,-33.49005365],
                                    rtol=1e-5).tolist(),
                          [True]*3)

        #Position of the Sun at the date of the test
        eclon=Angle.dec((+1,280,22,21.9))
        eclat=Angle.dec((-1,0,0,2.7))
        A,h=self.impact.ecl2loc(eclon*Angle.Deg,eclat*Angle.Deg)

        #Position of Betelgeuse
        eclon=Angle.dec((+1,88,45,16.6))
        eclat=Angle.dec((-1,16,1,37.2))
        A,h=self.impact.ecl2loc(eclon*Angle.Deg,eclat*Angle.Deg)
        self.assertAlmostEqual(A*Angle.Rad,57.27518638612843,5)
        self.assertAlmostEqual(h*Angle.Rad,-76.20677246845091,5)

        A=Angle.dec((+1,57,16,30.7))
        h=Angle.dec((-1,76,12,24.4))
        eclon,eclat=self.impact.loc2ecl(A*Angle.Deg,h*Angle.Deg)
        self.assertAlmostEqual(eclon*Angle.Rad,88.75461469860417,5)
        self.assertAlmostEqual(eclat*Angle.Rad,-16.027004471139914,5)        
       
    def test_moon_loc2bod(self):
        self.assertEqual(np.isclose(self.crater.Tloc2bod.flatten(),
                                    [-0.45982258,-0.71502441,0.52659594,
                                     -0.47029697,0.69909948 ,0.53859138,
                                      0.75324894,0.,0.65773554],
                                    rtol=1e-5).tolist(),
                          [True]*9)
    
    def test_moon_local(self):
        self.assertEqual(np.isclose(self.crater.posLocal,
                                    [0.,0.,0.],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.crater.velLocal,
                                    [0.,3.50340129,0.],
                                    rtol=1e-5).tolist(),
                          [True]*3)
    
    def test_moon_body(self):
        self.assertEqual(np.isclose(self.crater.posBody,
                                    [920173.74904705,941134.57557525,1149327.08234927],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.crater.velBody,
                                    [-2.50501745,2.44922603,0.],
                                    rtol=1e-5).tolist(),
                          [True]*3)

    def test_moon_ecliptic(self):
        self.crater.updateLocation(self.tdb_moon)
        self.assertEqual(np.isclose(self.crater.posEcl,
                                    [-189857.68536427,1287964.72727012,1165550.37669598],
                                    rtol=1e-5).tolist(),
                          [True]*3)
        self.assertEqual(np.isclose(self.crater.velEcl,
                                    [-3.47523194,-0.43509669,-0.08529044],
                                    rtol=1e-5).tolist(),
                          [True]*3)

    def test_moon_ecl2loc(self):
        self.crater.updateLocation(self.tdb_moon)
        eclon=25.3157371
        eclat=-1.2593327
        A,h=self.crater.ecl2loc(eclon*Angle.Deg,eclat*Angle.Deg)
        self.assertEqual(np.isclose([A*Angle.Rad,h*Angle.Rad],
                                    [255.7181,11.6614],
                                    atol=1e-2).tolist(),
                         [True]*2)

    def test_moon_loc2ecl(self):
        self.crater.updateLocation(self.tdb_moon)
        A=229.2705
        h=28.8062
        eclon,eclat=self.crater.loc2ecl(A*Angle.Deg,h*Angle.Deg)
        self.assertEqual(np.isclose([eclon*Angle.Rad,eclat*Angle.Rad],
                                    [55.1580499,-5.0588748],
                                    atol=1e-2).tolist(),
                         [True]*2)
    
    def timing_update(self):
        self.crater.updateLocation(self.tdb_moon)

if __name__=='__main__':
    #Testing
    unittest.main(argv=['first-arg-is-ignored'],exit=False)
    
    #"""
    #Timing
    print("Timing loadKernels:")
    get_ipython().magic('timeit -n 10 Test().timing_load_kernels()')
    #"""
    
    #Timing
    print("Timing update body:")
    t=Test()
    t.timing_update_earth()
    get_ipython().magic('timeit -n 100 t.earth.updateBody(0)')
    
    #Timing
    print("Timing update location:")
    get_ipython().magic('timeit -n 100 Test().timing_update()')

