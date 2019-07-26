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

#################################################################################                                                                                           
#Global variables
#################################################################################
import os
try:
    ROOTDIR=os.path.abspath(os.path.dirname(__file__))
except:
    ROOTDIR=os.path.abspath('')

#################################################################################                                                                                           
#This code is used only for development purposes                                                                                           
#################################################################################
from IPython.display import HTML, Image
import IPython.core.autocall as autocall

try:
    cfg=get_ipython().config
except NameError:
    def Image(url="",filename="",f=""):
        pass
    class get_ipython(object):
        def run_line_magic(self,x,y):
            pass
        def magic(self,command):
            pass

#################################################################################                                                                                           
#Test
#################################################################################
if __name__=="__main__":
    print(f"Root directory:{_ROOTDIR}")

