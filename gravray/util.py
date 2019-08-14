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
        exp=np.int(np.floor(np.log10(xa)))
        man=s*xa/10**(exp)
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
#CLASS JACOBIANS
#################################################################################
class Jacobians(object):
    """
    This abstract class contains useful methods for computing Jacobians.
    
    Attributes:
        None.
        
    Methods:
        computeJacobian: numerically compute jacobian.
    
    """
    def computeNumericalJacobian(jfun,x,dx,**args):
        """
        Computes numerically the Jacobian matrix of a multivariate function.
        
        Parameters:
            jfun: multivariate function with the prototype "def jfun(x,**args)", function
            x: indepedent variables, numpy array (N).
            dx: step size of independent variables, numpy array (N).
            **args: argument of the function
            
        Return:
            y: dependent variables, y=jfun(x,**args)
            Jyx: Jacobian matrix:
            
              Jif= [dy_1/dx_1,dy_1/dx_2,...,dy_1/dx_N,
                    dy_2/dx_1,dy_2/dx_2,...,dy_2/dx_N,
                                 . . . 
                    dy_N/dx_1,dy_N/dx_2,...,dy_N/dx_N,]
        """
        N=len(x)
        J=np.zeros((N,N))
        y=jfun(x,**args)
        for i in range(N):
            for j in range(N):
                pre=[x[k] for k in range(j)]
                pos=[x[k] for k in range(j+1,N)]
                yi=lambda t:jfun(pre+[t]+pos,**args)[i]
                dyidxj=(yi(x[j]+dx[j])-yi(x[j]-dx[j]))/(2*dx[j])
                J[i,j]=dyidxj
        return y,J

    def calcKeplerianJacobians(mu,celements,state):
        """
        Compute the Jacobian Matrix of the transformation from classical 
        orbital elements (a,e,i,w,W,M) to cartesian state vector (x,y,z,x',y',z').

        Return:

            Jc2k = [dx/da,dx/de,dx/di,dx/dw,dx/dW,dx/dM,
                    dy/da,dy/de,dy/di,dy/dw,dy/dW,dy/dM,
                    dz/da,dz/de,dz/di,dz/dw,dz/dW,dz/dM,
                    dx'/da,dx'/de,dx'/di,dx'/dw,dx'/dW,dx'/dM,
                    dy'/da,dy'/de',dy'/di,dy'/dw,dy'/dW,dy'/dM,
                    dz'/da,dz'/de,dz'/di',dz'/dw,dz'/dW,dz'/dM],

                    Numpy array 6x6, units compatible with mu and a.
        """
        a,e,i,W,w,M=celements
        q=a*(1-e)

        #Orbit signature
        if e<1:
            s=+1
        elif e>1:
            s=-1
        else:
            s=0
        
        #Trigonometric function
        cosi,sini=Angle.calcTrig(i)
        cosw,sinw=Angle.calcTrig(w)
        cosW,sinW=Angle.calcTrig(W)

        #Components of the rotation matrix
        A=(cosW*cosw-cosi*sinW*sinw);B=(-cosW*sinw-cosw*cosi*sinW)
        C=(cosw*sinW+sinw*cosi*cosW);D=(-sinw*sinW+cosw*cosi*cosW)
        F=sinw*sini;G=cosw*sini

        #Primary auxiliar variables
        ab=np.abs(a)
        n=np.sqrt(mu/ab**3)
        nu=n*a**2
        eps=np.sqrt(s*(1-e**2))

        #Get cartesian coordinates
        x,y,z,vx,vy,vz=state
        r=(x**2+y**2+z**2)**0.5
        nur=nu/r
        
        #Eccentric anomaly as obtained from indirect information
        #From the radial equation: r = a (1-e cos E)
        cosE=(1/e)*(1-r/a)

        #From the general equation for y
        #NOTE: This is the safest way to obtain sinE without the danger of singularities
        sinE=(y-a*(cosE-e)*C)/(ab*eps*D)

        #dX/da
        Ja=np.array([x/a,y/a,z/a,-vx/(2*a),-vy/(2*a),-vz/(2*a)])

        #dX/de
        dcosEde=-s*a*sinE**2/r
        dsinEde=a*cosE*sinE/r
        dnurde=(nu*a/r**2)*(cosE-(ab/r)*e*sinE**2)
        depsde=-s*e/eps

        drAde=a*(dcosEde-1)
        drBde=ab*(depsde*sinE+eps*dsinEde)

        dvAde=-(dnurde*sinE+nur*dsinEde)
        dvBde=(dnurde*eps*cosE+nur*depsde*cosE+nur*eps*dcosEde)

        Je=np.array([
            drAde*A+drBde*B,
            drAde*C+drBde*D,
            drAde*F+drBde*G,
            dvAde*A+dvBde*B,
            dvAde*C+dvBde*D,
            dvAde*F+dvBde*G,
        ])

        #dX/di
        Ji=np.array([z*sinW,-z*cosW,-x*sinW+y*cosW,vz*sinW,-vz*cosW,-vx*sinW+vy*cosW])

        #dX/dw
        Jw=np.array([-y*cosi-z*sini*cosW,x*cosi-z*sini*sinW,sini*(x*cosW+y*sinW),            -vy*cosi-vz*sini*cosW,vx*cosi-vz*sini*sinW,sini*(vx*cosW+vy*sinW)])

        #dX/dW
        JW=np.array([-y,x,0,-vy,vx,0])

        #dX/dM
        JM=np.concatenate(((ab**3/mu)**0.5*np.array([vx,vy,vz]),
                           (mu*ab**3)**0.5*np.array([-x/r**3,-y/r**3,-z/r**3])))

        #Jacobian
        Jck=np.array([Ja,Je,Ji,JW,Jw,JM]).transpose()

        return Jck

    def calcMapJacobian(elements,scales):
        """
        Parameters:
            epsilon: bound elements, numpy array (N)
            scales: scales for the bound elements ()

        Return:

            Jif= [dE_1/de_1,        0,        0,...,        0,
                          0,dE_2/de_2,        0,...,        0,
                          0,        0,dE_2/de_2,...,        0,
                                 . . . 
                          0,        0,        0,...,dE_N/de_N]

            where dE/de = (1/s) /[x(1-x)] and x = e/s.
        """
        JEe=np.identity(6)
        JeE=np.identity(6)
        for i,eps in enumerate(elements):
            x=eps/scales[i]
            JEe[i,i]=(1/scales[i])/(x*(1-x))
            JeE[i,i]=1/JEe[i,i]
            
        return JEe,JeE
    
    def calcImpactJacobian(body,Rimp,state):
        """
        Compute the Jacobian Matrix of the transformation from local impact conditions 
        (lon,lat,alt,A,h,v) to cartesian state vector (x,y,z,x',y',z') (in the body reference frame).

        Parameters:
            Rimp: Impact vector
                lon: Geographic longitude (0,2pi), float, radians
                lat: Geographic latitude (0,2pi), float, radians
                alt: Altitude over the ellipsoid (0,inf), float, km
                A: Azimuth (0,2pi), float, radians
                h: Elevation (-pi/2,pi/2), float, radians
                v: Impact speed (-inf,inf), float, km/s (negative if it is impacting)
            state: State vector (x,y,z,x',y',z'), numpy array (6)

        Return:

            Jcl = [dx/dlon,dx/dlat,dx/dalt,dx/dA,dx/dh,dx/dv,
                   dy/dlon,dy/dlat,dy/dalt,dy/dA,dy/dh,dy/dv,
                   dz/dlon,dz/dlat,dz/dalt,dz/dA,dz/dh,dz/dv,
                   dx'/dlon,dx'/dlat,dx'/dalt,dx'/dA,dx'/dh,dx'/dv,
                   dy'/dlon,dy'/dlat,dy'/dalt,dy'/dA,dy'/dh,dy'/dv,
                   dz'/dlon,dz'/dlat,dz'/dalt,dz'/dA,dz'/dh,dz'/dv],

                Numpy 6x6 array.
        """
        #Local to rotating
        lon,lat,alt,A,h,vimp=Rimp
        x,y,z,vx,vy,vz=state
        
        coslon,sinlon=Angle.calcTrig(lon)
        coslat,sinlat=Angle.calcTrig(lat)
        cosA,sinA=Angle.calcTrig(A)
        cosh,sinh=Angle.calcTrig(h)
        
        P=body.Prot
        a=body.Ra
        b=body.Rc
        Tbod2ecl=body.Tbod2ecl
        
        #Auxiliar
        fr=2*np.pi*np.sqrt(x**2+y**2)/(P*vimp)
        N=a**2/np.sqrt(a**2*coslat**2+b**2*sinlat**2)
        n2=(2*np.pi/P)**2

        #dX/dlon:
        Jlon=np.array([-y,x,0,-vy,vx,0])

        #dX/dlat:
        dxdlat=(a**2-b**2)*coslat*sinlat*N**3/a**4*coslat*coslon-(N+alt)*sinlat*coslon
        dydlat=(a**2-b**2)*coslat*sinlat*N**3/a**4*coslat*sinlon-(N+alt)*sinlat*sinlon
        Jlat=np.array([
            dxdlat,
            dydlat,
            b**2*(a**2-b**2)*coslat*sinlat*N**3/a**6*sinlat+(b**2*N/a**2+alt)*coslat,
            -vimp*cosh*cosA*coslat*coslon-n2*sinlon/(fr*vimp)*(x*dxdlat+y*dydlat)-vimp*sinh*sinlat*coslon,
            -vimp*cosh*cosA*coslat*sinlon+n2*coslon/(fr*vimp)*(x*dxdlat+y*dydlat)-vimp*sinh*sinlat*sinlon,
            vimp*(-cosh*cosA*sinlat+sinh*coslat)
        ])

        #dX/dalt:
        Jalt=np.array([
            coslat*coslon,coslat*sinlon,sinlat,
            -n2*sinlon/(fr*vimp)*(x*coslat*coslon+y*coslat*sinlon),
            +n2*coslon/(fr*vimp)*(x*coslat*coslon+y*coslat*sinlon),
            0
        ])

        #dX/dA:
        JA=np.array([0,0,0,
            vimp*(cosh*sinA*sinlat*coslon-cosh*cosA*sinlon),
            vimp*(cosh*sinA*sinlat*sinlon+cosh*cosA*coslon),
            -vimp*cosh*sinA*coslat,
           ])

        #dX/dh:
        Jh=np.array([0,0,0,
            vimp*(sinh*cosA*sinlat*coslon+sinh*sinA*sinlon+cosh*coslat*coslon),
            vimp*(sinh*cosA*sinlat*sinlon-sinh*sinA*coslon+cosh*coslat*sinlon),
            vimp*(-sinh*cosA*coslat+cosh*sinlat),
            ])

        #dX/dvimp:
        Jv=np.array([0,0,0,vx/vimp+sinlon*fr,vy/vimp-coslon*fr,vz/vimp])

        Jcl=np.array([Jlon,Jlat,Jalt,JA,Jh,Jv]).transpose()
        Jel=np.zeros_like(Jcl)
        for i in range(6):
            Jel[:3,i]=spy.mxv(Tbod2ecl,Jcl[:3,i])
            Jel[3:,i]=spy.mxv(Tbod2ecl,Jcl[3:,i])
        return Jcl,Jel

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

