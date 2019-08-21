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

# # GravRay Util Classes, Functions and Data

from gravray import *

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#################################################################################
#CLASS UTIL
#################################################################################
class Util(object):
    """
    This abstract class contains useful methods for the package.
    
    Attributes:
        None.
        
    Methods:
        fin2Inf:
        inf2Fin:
        genIndex:
        transformState:
        transformElements:
    """ 
    log=math.log
    log10=math.log10
    exp=math.exp
    sin=math.sin
    cos=math.cos
    asin=math.asin
    acos=math.acos
    sqrt=math.sqrt
    
    def fin2Inf(x,scale=1):
        """
        Map variable x from the interval (0,scale) to a new variable t in the interval (-inf,+inf).
        
        Parameters:
            x: value in the range (0,scale), float.

        Optional:
            scale: maximum value for x (default 1), float.
        
        Return:
            Mapped value t: x->0 correspond to t->-inf, x->scale t->+inf, float.        
        """
        u=x/scale
        try:
            t=Util.log(u/(1-u))
            return t
        except ValueError as e:
            errorMsg(e,f"x value ({x}) must be in the interval (0,scale) (eg. (0,{scale}))")
            raise 

    def inf2Fin(t,scale=1):
        """
        Map variable x from the interval (-inf,inf) to a new variable t in the interval (0,scale).
        
        Parameters:
            t: value in the range (-inf,inf), float.

        Optional:
            scale: maximum value for x (default 1), float.
        
        Return:
            Mapped value x: t->-inf correspond to x -> 0, t->+inf correspond to x -> 1, float.        
        """
        x=scale/(1+Util.exp(-t))
        return x
    
    def fin2Uno(x,scale=1):
        """
        Simple mapping from a finite interval (0,scale) to (0,1)
        
        Parameters:
            x: value in the range (0,scale), float
            
        Optional:
            scale: maximum value for x (default 1), float.
        
        Return:
            Mapped value t: t = x/scale, float.
        """
        return x/scale
    
    def uno2Fin(t,scale=1):
        """
        Simple mapping from a finite interval (0,1) to (0,scale)
        
        Parameters:
            t: value in the range (0,1), float
            
        Optional:
            scale: maximum value for x (default 1), float.
        
        Return:
            Mapped value t: x = t*scale, float.
        """
        return t*scale
    
    def genIndex(probs):
        """
        Given a set of (normalized) probabilities, randomly generate an index n following the 
        probabilities.

        For instance if we have 3 events with probabilities 0.1, 0.7, 0.2, genSample will generate
        a number in the set (0,1,2) having those probabilities, ie. 1 will have 70% of probability.
        
        Parameters:
            probs: Probabilities, numpy array (N), adimensional
                NOTE: It should be normalized, ie. sum(probs)=1
            
        Return:
            n: Index [0,1,2,... len(probs)-1], integer
            
        Example:
            genIndex([0.1,0.7,0.2])
        """
        cums=np.cumsum(probs)
        if cums[-1]!=1:
            raise ValueError("Probabilities must be normalized, ie. sum(probs) = 1")
        cond=(np.random.rand()-cums)<0
        isort=np.arange(len(probs))
        n=isort[cond][0] if sum(cond)>0 else isort[0]
        return n
    
    def transformState(state,factors,implicit=False):
        """
        Change units of a state vector.
        
        Parameters:
            state: state vector (x,y,z,vx,vy,vz), float (6), (L,L,L,L/T,L/T,L/T)
            [facLen,facVel]: convesion factors, float (2)
            
        Return:
            state: converted state vector x*facLen,y*facLen,z*facLen,z*facLen,vx*facVel,vy*facVel,vz*facVel
                    float(6),(L,L,L,L/T,L/T,L/T)
        """
        facLen,facVel=factors
        
        if implicit:
            state[:3]*=facLen
            state[3:]*=facVel
        else:
            return np.concatenate((state[:3]*facLen,state[3:]*facVel))

    def transformElements(elements,factors,implicit=False):
        """
        Change units of an elements vector

        Parameters:
            elements: elements vector (a,e,i,W,w,M), float (6), (L,1,RAD,RAD,RAD,RAD)
            [facLen,facAng]: convesion factors (length, angles), float (2)

        Return:
            elements: converted elements vector a*facLen,e,i*facAng,W*facAng,w*facAng,M*facAng
                    float(6),(L,L,L,L/T,L/T,L/T)
        """
        facLen,facAng=factors
        if implicit:
            elements[:1]*=facLen
            elements[2:]*=facAng
        else:
            return np.concatenate((elements[:1]*facLen,[elements[1]],elements[2:]*facAng))    

        
    def chunkList(mylist,chunksize):
        """
        Split a list in chunks with maximum size equal to chunksize
        
        Parameters:
            myslist: a list of objects, list.
            chunksize: size of each chunk, int.
        
        Return:
            chunks: iterator of the chunks corresponding to mylist. 
        
        Examples:
            Command:
                [sublist for sublist in Util.chunkList([1,2.3,"hola",np.int,3,4,5],3)]
            produces:
                [[1, 2.3, 'hola'], [int, 3, 4], [5]]
    
        """
        for i in range(0,len(mylist),chunksize):yield mylist[i:i+chunksize]
            
    def medHistogram(data,**args):
        """
        Compute 1d histogram.
        
        Parameters:
            data: data values, numpy array (N)
            **args: options for the numpy histogram routine, dictionary
            
        Return:
            h: histogram.
            xm: mid points of the intervals
        """
        h,x=np.histogram(data,**args)
        xm=(x[1:]+x[:-1])/2
        return h,xm
    
    def mantisaExp(x):
        """
        Calculate the mantisa and exponent of a number.
        
        Parameters:
            x: number, float.
            
        Return:
            man: mantisa, float
            exp: exponent, float.
            
        Examples:
            m,e=mantisaExp(234.5), returns m=2.345, e=2
            m,e=mantisaExp(-0.000023213), return m=-2.3213, e=-5
        """
        xa=np.abs(x)
        s=np.sign(x)
        try:
            exp=np.int(np.floor(np.log10(xa)))
            man=s*xa/10**(exp)
        except OverflowError as e:
            man=exp=0
        return man,exp
    
    def arcDistance(lon1,lat1,lon2,lat2):
        """
        Compute arc distance between two points.

        Parameters:
            lon1,lat1,lon2,lat2: latitude and longitudes of the points, float, radians.
            
        Return:
            arc: angle between points, float, radians
            
        """
        #Haversine
        HAV=lambda theta:Util.sin(theta/2)**2

        #Haversine
        h=HAV(lat2-lat1)+Util.cos(lat1)*Util.cos(lat2)*HAV(lon2-lon1)

        #Angular distance
        delta=2*Util.asin(Util.sqrt(h))

        return delta

#################################################################################
#CLASS ANGLE
#################################################################################
class Angle(object):
    """
    Abstract class containing angle related data and methods.
    
    Attributes:
        Deg: factor converting from degrees to radians.
        Rad: factor converting from radians to degrees.
        
    Methods:
        calcTrig: calculate the basic trigonometric function (cos, sin)
        dms: convert from decimal to sexagesimal.
        dec: convert from sexagesimal to decimal.
    """
    
    
    Deg=np.pi/180
    Rad=1/Deg
    
    def calcTrig(angle):
        """
        Calculate the basic trigonometric function (cos, sin)

        Parameters:
            angle: angle, float, radians.
        Return:
            cos(angle), sin(angle): common trig. functions, tuple (2)
        """
        return math.cos(angle),math.sin(angle)

    def dms(value):
        """
        Convert a decimal angle to the hexagesimal (d:m:s) format.
        
        Parameters:
            dec: Angle in decimal, float, degrees
        
        Return:
            dms: Angle in dms, tuple/list/array(4), (sign,deg,min,sec)
        """
        sgn=np.sign(value)
        val=np.abs(value)
        deg=np.floor(val)
        rem=(val-deg)*60
        min=np.floor(rem)
        sec=(rem-min)*60
        return (sgn,deg,min,sec)
    
    def dec(dms):
        """
        Convert an angle expressed in sexagesimal (d:m:s) to its decimal value.
        
        Parameters:
            dms: Angle in dms, tuple/list/array(4), (sign,deg,min,sec)
            
        Return:
            dec: Angle in decimal, float, degree
        """
        return dms[0]*(dms[1]+dms[2]/60.0+dms[3]/3600.0)

#################################################################################
#CLASS CONST
#################################################################################
class Const(object):
    """
    Abstract class containing constants and units.
    
    Attributes:
        au: Astronomica unit
        G: Gravitational constant
        Min,Hour,Day,Year,SideralMonth: Units of time
        
    Methods:
        calcTrig: calculate the basic trigonometric function (cos, sin)
        dms: convert from decimal to sexagesimal.
        dec: convert from sexagesimal to decimal.
    """
    #Astronomical unit
    au=1.4959787070000000e11 #km, value assumed in DE430
    aphelion=1.0167*au
    
    #Gravitational Constant
    G=6.67430e-11 #m^3/(kg s^2), Wikipedia
    
    #Common units of time
    s=1.0
    Min=60.0 
    Hour=60.0*Min 
    Day=24.0*Hour
    Year=365.24*Day
    SideralMonth=27.321661*Day
    
    #Common units of length
    km=1000.0 # m

from quadpy import ncube as quad
class MultiCube(object):
    """
    Multidimensional cube integration schemes.  Inspired on: https://pypi.org/project/quadpy/

    Initializacion attributes:
    
        multifunc: multidimensional integration routine, function with signature:
                    
                           func(X,**kwargs)
                           
                   where X is a matrix (NxM) with N the number of variables and M the number 
                   of values where the variables will be evaluated.
                   
        variables: list with name of variables, list of strings (N)
        
    Optional initialization attributes:
        
        nscheme: name of the integration scheme (see MultiCube._schema).
                   
    Examples: 
    
        def func(X,factor=1):
            r,q,f=X
            p=factor*1/np.sqrt((2*np.pi)**3)*np.exp(-r**2/2)*r**2*np.cos(q)
            return p
            
        nint=MultiCube(func,["r","q","f"],"stroud_cn_5_5")
        i=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))
        
        nint.setScheme("dobrodeev_1978")
        i=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))        
    """
    
    _schema = [
    "dobrodeev_1970",
    "dobrodeev_1978",
    "ewing",
    "hammer_stroud_1n",
    "hammer_stroud_2n",
    "mustard_lyness_blatt",
    "phillips",
    "stroud_1957_2",
    "stroud_1957_3",
    "stroud_1966_a",
    "stroud_1966_b",
    "stroud_1966_c",
    "stroud_1966_d",
    "stroud_1968",
    "stroud_cn_1_1",
    "stroud_cn_1_2",
    "stroud_cn_2_1",
    "stroud_cn_2_2",
    "stroud_cn_3_1",
    "stroud_cn_3_2",
    "stroud_cn_3_3",
    "stroud_cn_3_4",
    "stroud_cn_3_5",
    "stroud_cn_3_6",
    "stroud_cn_5_2",
    "stroud_cn_5_3",
    "stroud_cn_5_4",
    "stroud_cn_5_5",
    "stroud_cn_5_6",
    "stroud_cn_5_7",
    "stroud_cn_5_8",
    "stroud_cn_5_9",
    "stroud_cn_7_1",
    "thacher",
    "tyler"]
    
    def __init__(self,multifunc,variables,nscheme="dobrodeev_1978"):
        self.multifunc=multifunc
        self.variables={var:i for i,var in enumerate(variables)}
        self.dim=len(variables)
        self.nscheme=nscheme
        self.setScheme(nscheme)
        
    def setScheme(self,nscheme):
        if nscheme not in self._schema:
            raise AssertionError(f"Scheme not recognized.")
        self.nscheme=nscheme
        self.scheme=quad.__dict__[nscheme](self.dim)

    def integrate(self,variables,args=()):
        """
        Compute the integral in a given subdomain of the function variables.
        
        Parameters:
        
            variables: dictionary with variable values or ranges, dictionary.
        
        Optional parametes:
        
            args: arguments for the function.
            
        Example:
        
            def func(X,factor=1):
                r,q,f=X
                p=factor*1/np.sqrt((2*np.pi)**3)*np.exp(-r**2/2)*r**2*np.cos(q)
                return p
            
            nint=MultiCube(func,["r","q","f"],"stroud_cn_5_5")
            i=nint.integrate({"r":[0.0,1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))

            i=nint.integrate({"r":[1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]},args=(1.0,))

            i=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))
        """
        iconst=[]
        values=[]
        ivars=[]
        intervals=()
        svars=list(variables.keys())
        list.sort(svars,key=lambda s:self.variables[s])
        for v in svars:
            val=variables[v]
            if len(val)==1:
                iconst+=[self.variables[v]]
                values+=val
            else:
                ivars+=[self.variables[v]]
                intervals+=(val,)
        sdim=len(ivars)
        scheme=quad.__dict__[self.nscheme](sdim)
        def f(x):
            M=x.shape[1]
            self.M=M
            X=np.zeros((self.dim,M))
            X[ivars,:]=x
            X[iconst,:]=np.array(list(values)*M).reshape(M,len(iconst)).transpose()
            p=self.multifunc(X,*args)
            return p
        i=scheme.integrate(f,quad.ncube_points(*intervals))
        return i

from scipy import integrate
from functools import partial

class _NQuad(object):
    
    def __init__(self,func,ranges,integrator=integrate.fixed_quad,opts=None):
        self.abserr = 0
        self.func = func
        self.ranges = ranges
        self.maxdepth = len(ranges)
        if opts is None:
            self.opts=[dict()]*self.maxdepth
        else:
            self.opts = opts
        self.integrator=integrator

    def integrate(self,*args,**kwargs):
        depth = kwargs.pop('depth', 0)
        
        ind = -(depth + 1)
        low, high = self.ranges[ind]
        opt = self.opts[ind]

        if depth + 1 == self.maxdepth:
            f = self.func
        else:
            f = partial(self.integrate,depth=depth+1)
            
        quad_r = self.integrator(f,low,high,args=args,**opt)

        try:
            value = quad_r[0]
            abserr = quad_r[1]
        except:
            value = quad_r
            abserr = None
        
        if abserr is None:
            self.abserr = None
        else:
            self.abserr = max(self.abserr, abserr)
                    
        if depth>0:
            return value
        else:
            return value,self.abserr
            
class MultiQuad(object):
    """
    Multidimensional quadrature integration.

    Initializacion attributes:
    
        multifunc: multidimensional integration routine, function with signature:
                    
                           func(X,**kwargs)
                           
                   where X is a matrix (NxM) with N the number of variables and M the number 
                   of values where the variables will be evaluated.
                   
        variables: list with name of variables, list of strings (N)
        
    Optional initialization attributes:
        
        integrator: name of the integrator (see MultiQuad._integrators).
                   
    Examples: 
    
        def func(X,factor=1):
            r,q,f=X
            p=factor*1/np.sqrt((2*np.pi)**3)*np.exp(-r**2/2)*r**2*np.cos(q)
            return p
            
        nint=MultiQuad(func,["r","q","f"],"quad")
        i=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))
        
        nint.setIntegrator("fixed_quad")
        i=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))        
    """    
    _integrators=[
        "fixed_quad",
        "quad",
        "romberg"
    ]

    def __init__(self,multifunc,variables,integrator="fixed_quad",opt=dict()):
        self.multifunc=multifunc
        self.variables={var:i for i,var in enumerate(variables)}
        self.dim=len(variables)
        self.opt=opt
        self.setIntegrator(integrator)
        self.fun_calls=0
        
    def setIntegrator(self,nintegrator):
        self.nintegrator=nintegrator
        if self.nintegrator in self._integrators:
            self.integrator=integrate.__dict__[self.nintegrator]
        else:
            raise AssertionError(f"Integrator {nintegrator} not recognized")

    def integrate(self,variables,args=()):
        """
        Compute the integral in a given subdomain of the function variables.
        
        Parameters:
        
            variables: dictionary with variable values or ranges, dictionary.
        
        Optional parametes:
        
            args: arguments for the function.
            
        Example:
        
            def func(X,factor=1):
                r,q,f=X
                p=factor*1/np.sqrt((2*np.pi)**3)*np.exp(-r**2/2)*r**2*np.cos(q)
                return p

            nint=MultiQuad(func,["r","q","f"],"quad")

            i=nint.integrate({"r":[0.0,1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))

            i=nint.integrate({"r":[1.0],"q":[-np.pi/2,np.pi/2],"f":[0.0,2*np.pi]},args=(1.0,))

            i=nint.integrate({"r":[1.0],"q":[np.pi/3],"f":[0.0,2*np.pi]},args=(1.0,))
        """
        
        iconst=[]
        values=[]
        ivars=[]
        intervals=[]
        svars=list(variables.keys())
        list.sort(svars,key=lambda s:self.variables[s])
        for v in svars:
            val=variables[v]
            if len(val)==1:
                iconst+=[self.variables[v]]
                values+=list(val)
            else:
                ivars+=[self.variables[v]]
                intervals+=[val]
        sdim=len(ivars)
        opts=[self.opt]*sdim

        if self.nintegrator in ["fixed_quad"]:
            def f(*x):
                M=len(x[0])
                self.M=M
                self.fun_calls+=M
                X=np.zeros((self.dim,M))
                X[ivars,:]=np.vstack(x)
                X[iconst,:]=np.array(list(values)*M).reshape(M,len(iconst)).transpose()
                p=self.multifunc(X,*args)
                return p
        else:
            def f(*x):
                self.M=1
                self.fun_calls+=1
                X=np.zeros((self.dim,1))
                X[ivars,:]=np.array([x]).transpose()
                X[iconst,:]=np.array(list(values)*self.M).reshape(self.M,len(iconst)).transpose()
                p=self.multifunc(X,*args)
                return p
            
        pint=_NQuad(f,intervals,integrator=self.integrator,opts=opts)
        i=pint.integrate()
        return i

