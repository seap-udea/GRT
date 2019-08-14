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

# # Test of GravRay Plot

from gravray import *
from gravray.plot import *
from gravray.stats import *

get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'FILE="\' + IPython.notebook.notebook_name + \'"\')')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# # Test Plot Class
# 
# Test suite of the Util submodule of GravRay.

TIMING=0
TEST=1

import unittest
class Test(unittest.TestCase):

    def timing_calc_trig(self):
        pass

    #"""START COMMENT
    def test_map_surface(self):
        fig=plt.figure(figsize=(8,4),constrained_layout=True)
        ax=fig.gca()
        fig.clf()
        ax=fig.gca()
        m=Map("surface",ax,projection="robin")
        m.setDecoration()
        fig.savefig(f"/tmp/surface.png")
        
    def test_map_sky(self):
        fig=plt.figure(figsize=(6,6),constrained_layout=True)
        ax=fig.gca()
        m=Map("sky",ax)
        m.drawmapboundary.update(dict(fill_color=None))
        m.setDecoration()
        fig.savefig(f"/tmp/sky.png")
        
    def test_plot_grid(self):
        #We have a distribution of points in N-d
        weights=[0.6]
        locs=[
            [0.5,0.5,-2.0],
            [2.0,0.3,-2.6]
        ]
        scales=[
            [1.3,0.7,0.5],
            [0.4,0.9,1.6]
        ]
        angles=[
            [-40.0*Angle.Deg,-86.0*Angle.Deg,0.0*Angle.Deg],
            [+80.0*Angle.Deg,-109.0*Angle.Deg,0.0*Angle.Deg]
        ]
        ranges=[
            [0.0,1.0],
            [0.0,1.0]
        ]
        mnd=MultiVariate([1,1,1,0,0])
        mnd.setUnflatten(weights,locs,scales,angles,ranges)
        ps=mnd.rvs(1000)

        properties=dict(
            q=dict(label=r"$q$",range=None),
            e=dict(label=r"$e$",range=None),
            i=dict(label=r"$i$ (deg)",range=None),
            W=dict(label=r"$\Omega$ (deg)",range=None),
            w=dict(label=r"$\omega$ (deg)",range=None),
        )
        G=PlotGrid(properties,figsize=2)
        hist=G.plotHist(ps)
        G.fig.savefig(f"/tmp/plotgrid.png")
    #"""
    #END COMMENT

if __name__=='__main__':
    #Testing
    if TEST:unittest.main(argv=['first-arg-is-ignored'],exit=False)
    
    #Timing Util
    if TIMING:
        print("Timing test:")
        get_ipython().magic('timeit -n 1000 Test().timing_fin2inf()',scope=globals())

