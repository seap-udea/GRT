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

# # GravRay Master File
# 
# This is the master file

#################################################################################
#External basic modules
#################################################################################
import spiceypy as spy
import numpy as np
import math
import unittest
from copy import deepcopy
from time import time,strftime

#################################################################################
#This code is used only for development purposes in Jupyter
#################################################################################
"""
For the developer:
    The purpose of the get_ipython class is to provide some response in the python 
    script resulting from the conversion of this notebook.
    
    If you want to add another IPyhton function resulting from a magic command to the class, 
    please verify in the resulting python script the corresponding IPython command.
    
    For instance, the magic "%matplotlib nbagg" is converted into:
    
        get_ipython().magic('matplotlib nbagg',globals())
        
    So, the routinge "magic" should be add to the get_ipython() class.
        
"""
from IPython.display import HTML, Image
import IPython.core.autocall as autocall
from IPython import get_ipython

try:
    cfg=get_ipython().config
except AttributeError:
    def Image(url="",filename="",f=""):
        pass
    class get_ipython(object):
        def run_line_magic(self,x,y):
            pass
        def run_cell_magic(self,x,y,z):
            pass
        def magic(self,command,scope=globals()):
            import re
            if "timeit" in command:
                s=re.search("timeit\s+-n\s+(\d+)\s+(.+)",command)
                n=int(s.group(1))
                expr=s.group(2)
                timeIt(expr,scope=scope,n=n)

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

#################################################################################
#Global variables
#################################################################################
#Get name of script including path (FILE) and directory where the script is located (ROOTDIR)
import os
try:
    FILE=__file__
    ROOTDIR=os.path.abspath(os.path.dirname(FILE))
except:
    import IPython
    ROOTDIR=os.path.abspath('')

#Stores the time of start of the script when gravray is imported
TIMESTART=time()
#Stores the time of the last call of elTime
TIME=time()
#Stores the duration between elTime consecutive calls 
DTIME=-1
DUTIME=[]

#################################################################################
#Common routines
#################################################################################
def tUnit(t):
    for unit,base in dict(d=86400,h=3600,min=60,s=1e0,ms=1e-3,us=1e-6,ns=1e-9).items():
        tu=t/base
        if tu>1:break
    return tu,unit,base

def elTime(verbose=1,start=False):
    """
    Compute the time elapsed since last call of this routine.  The displayed time 
    is preseneted in the more convenient unit, ns (nano seconds), us (micro seconds), 
    ms (miliseconds), s (seconds), min (minutes), h (hours), d (days)
    
    Parameters: None.
    
    Optional:
        verbose: show the time in screen (default 1), integer or boolean.
        start: compute time from program start (deault 0), integer or boolean.
        
    Return: None.
    
    Examples:
        elTime(), basic usage (show output)
        elTime(0), no output
        elTime(start=True), measure elapsed time since program 
        print(DTIME,DUTIME), show values of elapsed time
    """
    global TIMESTART,TIME,DTIME,DUTIME
    t=time()
    dt=t-TIME
    if start:
        dt=t-TIMESTART    
        msg="since script start"
    else:
        msg="since last call"
    dtu,unit,base=tUnit(dt)
    if verbose:print("Elapsed time %s: %g %s"%(msg,dtu,unit))
    DTIME=dt
    DUTIME=[dtu,unit]
    TIME=time()
    
def timeIt(expr,scope=globals(),n=10):
    """
    Timing function. It imitates the behavior of %timeit magic function

    Parameters:
        expr: Expression to execute, string.
    
    Optional: 
        n: number of exectutions
        
    Return:
        Time of execution as (time in seconds, (time in unit, unit, base of unit)), tuple
        
    Example:
        def f(x):
            suma=0
            for n in range(10):
                suma+=np.cosh(x**(n/10.))*np.log10(x**(n/10.))
            return suma

        timeIt("f(3)",n=1000)
            (4.2202472686767576e-05, (42.20247268676758, 'us', 1e-06))
    """
    try:
        exec(expr,scope)
    except Exception as inst:
        print(f"I could not execute expression:\n{expr}\nError:\n{inst}")
        return

    r=range(n)
    dt=0
    cmd="elTime(0);"+expr+";elTime(0)"
    for i in r:
        dts=[]
        exec(cmd,scope);dts+=[DTIME];
        exec(cmd,scope);dts+=[DTIME];
        exec(cmd,scope);dts+=[DTIME];
        exec(cmd,scope);dts+=[DTIME];
        exec(cmd,scope);dts+=[DTIME];
        dt+=min(dts)
    texec=dt/n
    tuexec=tUnit(texec)
    print(f"{n} loops, best of 5: {tuexec[0]} {tuexec[1]} per loop")
    return texec,tuexec

def errorMsg(error,msg):
    """
    Add a custom message msg to an error handle.
    
    Parameters:
        error: error handle, handle (eg. except ValueError as error)
        msg: message to add to error, string.
    
    Return: None.
    """
    error.args=(error.args if error.args else tuple())+(msg,)
    
def stop():
    raise AssertionError("Stop")

#################################################################################
#Program test
#################################################################################
if __name__=="__main__":
    print(f"Root directory:{ROOTDIR}")
    print(f"File:{FILE}")
    elTime(start=True)
    print("Timing test:")
    get_ipython().magic('timeit -n 1000 np.log10(np.pi)',scope=globals())

