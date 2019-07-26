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

class Util(object):
    
    def fin2Inf(x,scale=1):
        """
        Map variable x from the interval [0,scale] to a new variable t in the interval [-inf,+inf].
        x = 0 correspond to t->-inf
        x = 1 correspond to t->+inf
        """
        u=x/scale
        t=np.log(u/(1-u))
        return t

    def inf2Fin(t,scale=1):
        """
        Map variable t from the interval (-inf,inf) to a new variable x in the interval [0,scale].
        t->-inf correspond to x = 0
        t->+inf correspond to x = 1
        """
        x=scale/(1+np.exp(-t))
        return x
    
    def genIndex(probs):
        """
        Given a set of (normalized) randomly generate the index n following the probabilities.
        For instance if we have 3 events with probabilities 0.1, 0.7, 0.2, genSample will generate
        a number in the set (0,1,2) with those probabilities.
        
        Parameters:
            probs: Probabilities, numpy array (N), adimensional
            NOTE: It should be normalized, ie. sum(probs)=1
            
        Return:
            n: Index [0,1,2,... len(probs)-1], integer
        """
        cond=(np.random.rand()-np.cumsum(probs))<0
        isort=np.arange(len(probs))
        n=isort[cond][0] if sum(cond)>0 else isort[0]
        return n

class Angle(object):
    Deg=np.pi/180
    Rad=1/Deg
    
    def calcTrig(angle):
        """
        Parameters:
            angle: angle, float, radians
        Return:
            cos(angle), sin(angle): common trig. functions, tuple (2), adimensiona
        """
        return np.cos(angle),np.sin(angle)

    def dms(value):
        """
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
        Parameters:
            dms: Angle in dms, tuple/list/array(4), (sign,deg,min,sec)
        Return:
            dec: Angle in decimal, float, degree
        """
        return dms[0]*(dms[1]+dms[2]/60.0+dms[3]/3600.0)

class Const(object):
    #Astronomical
    au=1.4959787070000000e8 #km, value assumed in DE430
    
    #Time
    Min=60.0 # seconds
    Hour=60.0*Min
    Day=24.0*Hour
    Year=365.24*Day
    SideralMonth=27.321661*Day
    
    #Length
    km=1000.0 # m
    au=1.4959787070000000e8*km
    
    #Speed
    kms=1000.0 # m/s
    
    #Units transformation
    def transformState(state,factors,implicit=False):
        """
        Change units of a state vector 
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

    #Orbital elements
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

