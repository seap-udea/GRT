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

# # GRT Parallel Processing
# 

from gravray import *
from gravray.util import *
from gravray.sampling import *
from gravray.spice import *
from gravray.orbit import *
from gravray.stats import *

from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from itertools import product as cartesian

def chunks(l, n):
    for i in range(0, len(l), n):yield l[i:i+n]

Spice.loadKernels()
NP=mp.cpu_count()
print("Number of processors: ",NP)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#import ray
#np=psutil.cpu_count(logical=False)
#ray.init(num_cpus=2)

# ## Common data

body="EARTH"
earth=Body(body)

# ## Parallel function

def rayProcessing(initial):
    t=initial[0]
    site=initial[1]
    direction=initial[2]
    ray=GrtRay(site,direction[0][0],direction[0][1],direction[1])
    ray.updateRay(t)
    try:
        ray.propagateRay(t)
        detJ=ray.calcJacobianDeterminant()
    except AssertionError as e:
        detJ=0
    raydf=ray.packRay()
    raydf["detJ"]=detJ
    return raydf
    
def rayProcessingMulti(initials):
    print(f"Processing {len(initials)} initial conditions")
    raydfs=[rayProcessing(initial) for initial in initials]
    return raydfs

# ## Test data

ts=[Spice.str2t("02/15/2013 03:20:34 UTC")]
siteprops=[[61.1*Angle.Deg,54.8*Angle.Deg,23.3*Const.km]]
sites=[]
for siteprop in siteprops:
    sites+=[Location(earth,siteprop[0],siteprop[1],siteprop[2])]
directions=[[[101.1*Angle.Deg,15.9*Angle.Deg],-18.6*Const.km/Const.s]]

#List of conditions
initials=list(cartesian(*[ts,sites,directions]))
rayProcessingMulti(initials)

# ## Massive input data

#Numbers
Ntimes=5
Nsites=10
Npoints=5
Nvels=5

#Times
print("Preparing times...")
tini=Spice.str2t("02/15/2013 03:20:34")
tend=tini+Const.Year
ts=np.linspace(tini,tend,10)

#Sites
print("Preparing sites...")
elTime(0)
H=23.3*Const.km
points=Sample(Nsites)
points.genUnitSphere()
siteprops=np.zeros((Nsites,3))
siteprops[:,:2]=points.pp[:,1:]
siteprops[:,2]=H*np.ones(Nsites)
sites=[]
for siteprop in siteprops:
    sites+=[Location(earth,siteprop[0],siteprop[1],siteprop[2])]
elTime()

#Directions
print("Preparing directions...")
elTime(0)
gpoints=Sample(Npoints)
gpoints.genUnitHemisphere()
speeds=-np.linspace(11.2,72.0,Nvels)*Const.km/Const.s
directions=list(cartesian(*[gpoints.pp[:,1:].tolist(),speeds]))
elTime()

#Initial conditions
print("Preparing initial conditions...")
elTime(0)
initials=list(cartesian(*[ts,sites,directions]))
elTime()

Ninitials=len(initials)
print(f"Number of initial conditions: {Ninitials} = {len(ts)}(ts)*{len(sites)}(sites)*{len(directions)}(dirs)")

print(f"Sequential processing of {Ninitials} rays:")
dt,dtu=elTime(0)
tinitials=initials[:10]
rays=rayProcessingMulti(tinitials)
dt,dtu=elTime()
tpray=dt/len(tinitials)
tupray=tUnit(tpray)
totrays=tpray*Ninitials
toturays=tUnit(totrays)
print(f"Total duration: {dtu[0]} {dtu[1]}, Duration per ray: {tupray[0]} {tupray[1]}")
print(f"Estimated total: {toturays[0]} {toturays[1]}")

# ## Prepare chunks

npchunk=np.int(np.ceil(Ninitials/NP))
cinitials=[initial for initial in chunks(initials,npchunk)]
Nchunks=len(cinitials)
print(f"{Nchunks} chunks containing {npchunk} initial conditions")
tchunk=tpray*npchunk
tchunku=tUnit(tchunk)
print(f"Estimated time per chunk: {tchunku[0]} {tchunku[1]}")

# ## Parallel processing

allraydfs=pd.DataFrame()
def joinResults(raydfs):
    global allraydfs
    allraydfs=pd.concat(((allraydfs,)+tuple(raydfs)))

pool=mp.Pool(NP)
elTime(0)
R=[pool.apply_async(rayProcessingMulti,args=(initials,),callback=joinResults) for initials in cinitials]
pool.close()
pool.join()
elTime()

print("Number of results:",len(allraydfs))

allraydfs.to_csv("rays_parallel.csv")

