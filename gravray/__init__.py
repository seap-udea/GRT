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

import spiceypy as spy
import numpy as np
import unittest
from copy import deepcopy
from time import time,strftime

#################################################################################                                                                                           
#Global variables
#################################################################################
import os
try:
    ROOTDIR=os.path.abspath(os.path.dirname(__file__))
except:
    ROOTDIR=os.path.abspath('')

TIME=time()

#################################################################################                                                                                           
#This code is used only for development purposes                                                                                           
#################################################################################
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
        def magic(self,command):
            pass

def elTime(verbose=1):
    """
    Compute the time elapsed since last call.
    """
    global TIME
    t=time()
    dt=t-TIME
    if verbose:print("Time elapsed: %g s, %g min, %g h"%(dt,dt/60.0,dt/3600.0))
    TIME=t
    return dt

#################################################################################                                                                                           
#Test
#################################################################################
if __name__=="__main__":
    print(f"Root directory:{ROOTDIR}")

