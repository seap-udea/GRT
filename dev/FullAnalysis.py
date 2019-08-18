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

# # GRT Full Analysis
# 

from gravray import *
from gravray.util import *
from gravray.stats import *
from gravray.spice import *
from gravray.plot import *
from gravray.sampling import *
from gravray.orbit import *

from tqdm import tqdm
from matplotlib.colors import LogNorm

from sys import argv

get_ipython().magic('matplotlib nbagg')

dt=elTime(0)
Spice.loadKernels()
cneos=pd.read_csv("data/cneos_fireball_data_location.csv")
cneos.set_index("Name",inplace=True)

cneos.head(50)

# ## Location

body="EARTH"
earth=Body(body)

#argv[1]="CNEOS-2018-07-25"
site=argv[1]

if re.search("CNEOS-",site):
    print("CNEOS Site")
    impact=cneos.loc[site]
    lon=impact["lon"]*Angle.Deg
    lat=impact["lat"]*Angle.Deg
    alt=impact["Altitude (km)"]*Const.km
    Aloc=impact["Aloc"]*Angle.Deg
    hloc=impact["hloc"]*Angle.Deg
    vimp=impact["vimp"]*Const.km/Const.s
    location=Location(earth,lon,lat,alt)
    datestring=impact["Ephemeris Date"].replace(" UTC","")
    fmt="%m/%d/%Y %H:%M:%S"

if site=="Bering_CNEOS":
    lon=172.4*Angle.Deg #rad
    lat=56.9*Angle.Deg #rad
    alt=25.6*Const.km #m
    vbod=np.array([6.3,-3.0,-31.2])*Const.km/Const.s
    location=Location(earth,lon,lat,alt)
    Aloc,hloc,vimp=location.vbod2loc(-vbod)
    datestring="2018-12-18 23:48:20"
    fmt="%Y-%m-%d %H:%M:%S"

if site=="Chelyabinsk_CNEOS":
    lon=61.1**Angle.Deg #rad
    lat=54.8*Angle.Deg #rad
    alt=23.3*Const.km #m
    vbod=np.array([+12.8,-13.3,-2.4])*Const.km/Const.s
    location=Location(earth,lon,lat,alt)
    Aloc,hloc,vimp=location.vbod2loc(-vbod)
    datestring="2013-02-15 03:20:33"
    fmt="%Y-%m-%d %H:%M:%S"

if site=="Chelyabinsk":
    lon=59.8703**Angle.Deg #rad
    lat=55.0958*Angle.Deg #rad
    alt=23.3*Const.km #m
    vimp=-18.6*Const.km/Const.s
    Aloc=103.5*Angle.Deg
    hloc=18.55*Angle.Deg
    location=Location(earth,lon,lat,alt)
    datestring="02/15/2013 03:20:34"
    fmt="%m/%d/%Y %H:%M:%S"

if site=="Viñales":
    lon=-83.8037*Angle.Deg #deg
    lat=+22.8820*Angle.Deg #deg
    alt=70.0*Const.km #m
    vimp=-16.9*Const.km/Const.s
    Aloc=178.9*Angle.Deg
    hloc=31.8*Angle.Deg
    location=Location(earth,lon,lat,alt)
    datestring="02/01/2019 18:17:10"
    fmt="%m/%d/%Y %H:%M:%S"

#Input conditions
print(f"Location: {site}")
print(f"\tDate: {datestring}")
print(f"\tlon. {lon*Angle.Rad:.4g}, lat. {lat*Angle.Rad:.4g}, alt. {alt/Const.km:.4g} km")
print(f"\tAloc = {Aloc*Angle.Rad:.4g}, hloc = {lat*Angle.Rad:.4g}, vimp = {vimp/Const.km} km/s")

#Common derivative
pref=f"siteanalysis-{site}"
tdb=Spice.str2tdb(datestring)
ray=GrtRay(location,Aloc,hloc,vimp)
ray.updateRay(tdb)
ray.propagateRay(tdb)
ray.terminal.calcUelements([Const.au,1,np.pi,2*np.pi,2*np.pi,2*np.pi])
locelements=ray.terminal.uelements
print(f"Terminal elements for {site}:",Util.transformElements(ray.terminal.elements,[1/Const.au,Angle.Rad]))
print("Unbound terminal elements:",locelements)

fig=plt.figure(figsize=(8,4))
ax=fig.gca()
m=Map("surface")
m.drawmeridians.update(dict(fontsize=8))
m.drawparallels.update(dict(fontsize=8))
m.date=datetime.strptime(datestring,fmt)
m.drawMap(ax)

p=m.plotMap(lon*Angle.Rad,lat*Angle.Rad,marker='x',ms=10,color='b')
t=m.textMap(lon*Angle.Rad,lat*Angle.Rad,site,ha="center",fontsize=12)
fig.savefig(f"figures/{pref}-map-location.png")

# ## Generate directions in the sky

sample=Sample(100)
sample.genUnitHemisphere()
sample.purgeSample()
As=sample.pp[:,1]
hs=sample.pp[:,2]

fig=plt.figure(figsize=(4,4))
ax=fig.gca()
m=Map("sky")
m.drawMap(ax)

s=m.scatterMap(As*Angle.Rad,hs*Angle.Rad)

# ## Compute rays

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
    [Util.fin2Inf(1e-2,360.0),Util.fin2Inf(360.0-1e-2,360.0)],
    [Util.fin2Inf(1e-2,360.0),Util.fin2Inf(360.0-1e-2,360.0)]
]
mnd=MultiVariate([1,1,1,0,0])
mnd.setUnflatten(weights,locs,scales,angles,ranges)

rays=pd.DataFrame()
for A,h in tqdm(zip(As,hs)):
    ray=GrtRay(location,A,h,vimp)
    ray.updateRay(tdb)
    try:
        ray.propagateRay(tdb)
        #J = |dEhel/dRimp| = |dchel/dRimp| x |dehel/dchel| x |dEhel/dehel| 
        detJ=ray.calcJacobianDeterminant()*             (1-ray.terminal.elements[1])*             Jacobians.calcDetMapJacobian(ray.terminal.elements,
                                          [Const.au,1,np.pi,2*np.pi,2*np.pi,2*np.pi])[0]
        if ray.terminal.elements[1]<1:
            ray.terminal.calcUelements(maxvalues=[1.02*Const.au,1,np.pi,2*np.pi,2*np.pi,2*np.pi])
            x=ray.terminal.uelements[:5]
            ph=mnd.pdf(x)
        else:
            ph=0
    except AssertionError as e:
        detJ=0
        ph=0
    raydf=ray.packRay()
    raydf["detJ"]=np.abs(detJ)
    raydf["ph"]=ph
    raydf["pi"]=ph*np.abs(detJ)
    rays=pd.concat((rays,raydf))
rays.reset_index(inplace=True)

# ## Visualize orbits in Elements Space

Util.log=np.log

rays_bound=rays[rays["e"]<=1]
rays_bound["Q"]=Util.fin2Inf(rays_bound["q"])
rays_bound["E"]=Util.fin2Inf(rays_bound["e"])
rays_bound["I"]=Util.fin2Inf(rays_bound["i"],180)
rays_bound["O"]=Util.fin2Inf(rays_bound["W"],360)
rays_bound["P"]=Util.fin2Inf(rays_bound["w"],360)

rays_bound[["q","e","i","W","w"]].describe()

elements=rays_bound[["Q","E","I","O","P"]]
elements.describe()

ps=mnd.rvs(10000)
properties=dict(
    Q=dict(label=r"$Q$",range=(-4,4)),
    E=dict(label=r"$E$",range=(-3,3)),
    I=dict(label=r"$I$",range=(-7,7)),
    O=dict(label=r"$O$",range=(-7,7)),
    P=dict(label=r"$P$",range=(-7,7)),
)
G=PlotGrid(properties,figsize=2)
h=G.plotHist(ps,bins=30)

s=G.scatterPlot(elements.values,color='r',marker='+')
s=G.scatterPlot(np.array([locelements[:5]]),color='w',marker='v',s=50)

G.fig.savefig(f"figures/{pref}-orbital-footprint.png")

# ## Visualize probability and Jacobian

rays["1/|log10(detJ)|"]=1/np.abs(np.log10(rays["detJ"]))
rays["1/|log10(ph)|"]=1/np.abs(np.log10(rays["ph"]))
rays["1/|log10(pi)|"]=1/np.abs(np.log10(rays["pi"]))

fig=plt.figure(figsize=(6,6))
ax=fig.gca()
m=Map("sky")
m.drawMap(ax)
scs=[]
ts=[]
qp=0

if qp:
    for s in scs:s.remove() 
    for t in ts:t.remove() 
    scs=[];ts=[];qp=0

scs+=[m.scatterMap(rays["A"].values,rays["h"].values,c=rays["1/|log10(detJ)|"].values)]
scs+=[m.scatterMap(Aloc*Angle.Rad,hloc*Angle.Rad,color='w',s=100,marker='v')]
for ind in rays.index:
    A=rays.loc[ind]["A"]
    h=rays.loc[ind]["h"]
    detJ=rays.loc[ind]["detJ"]
    ts+=[m.textMap(A,h,f"{detJ:.1e}",fontsize=6,color='w')]
qp=1

fig.savefig(f"figures/{pref}-sky-jacobian.png")

fig=plt.figure(figsize=(6,6))
ax=fig.gca()
m2=Map("sky")
m2.drawMap(ax)
scs2=[]
ts2=[]
qp2=0

if qp2:
    for s in scs2:s.remove() 
    for t in ts2:t.remove() 
    scs2=[];ts2=[];qp2=0

scs2+=[m2.scatterMap(rays["A"].values,rays["h"].values,c=rays["1/|log10(ph)|"].values)]
scs2+=[m2.scatterMap(Aloc*Angle.Rad,hloc*Angle.Rad,color='w',s=100,marker='v')]
for ind in rays.index:
    A=rays.loc[ind]["A"]
    h=rays.loc[ind]["h"]
    phm,phe=Util.mantisaExp(rays.loc[ind]["ph"])
    ts2+=[m2.textMap(A,h,f"{phm:.1f}({phe})",fontsize=6,color='w',ha='center')]
qp2=1

fig.savefig(f"figures/{pref}-sky-porb.png")

fig=plt.figure(figsize=(6,6))
ax=fig.gca()
m3=Map("sky")
m3.drawMap(ax)
scs3=[]
ts3=[]
qp3=0

if qp3:
    for s in scs3:s.remove() 
    for t in ts3:t.remove() 
    scs3=[];ts3=[];qp=0

scs3+=[m3.scatterMap(rays["A"].values,rays["h"].values,c=rays["1/|log10(pi)|"].values,marker='o')]
scs3+=[m3.scatterMap(Aloc*Angle.Rad,hloc*Angle.Rad,color='w',s=100,marker='v')]
for ind in rays.index:
    A=rays.loc[ind]["A"]
    h=rays.loc[ind]["h"]
    pim,pie=Util.mantisaExp(rays.loc[ind]["pi"])
    ts3+=[m3.textMap(A,h,f"{pim:.1f}({pie})",fontsize=6,color='w',ha='center')]
qp3=1

fig.savefig(f"figures/{pref}-sky-pimp.png")

# ## Contours

m=Map("sky")

nAs=nhs=20
nAs,nhs,As,hs,val=m.makeGrid(nAs,nhs)

detJs=np.zeros_like(val)
phs=np.zeros_like(val)
pis=np.zeros_like(val)
dirs=[]
for i in tqdm(range(nhs)):
    for j in range(nAs):
        h=hs[i,j]
        A=As[i,j]
        if h<0:
            detJ=0
            ph=0
        else:
            for v in [vimp]:
            #for vimp in np.linspace(11.1,43.0,10):
                #ray=GrtRay(location,A*Angle.Deg,h*Angle.Deg,-18.6*Const.km/Const.s)
                ray=GrtRay(location,A*Angle.Deg,h*Angle.Deg,v)
                ray.updateRay(tdb)
                try:
                    ray.propagateRay(tdb)
                    #J = |dEhel/dRimp| = |dchel/dRimp| x |dehel/dchel| x |dEhel/dehel| 
                    detJ=ray.calcJacobianDeterminant()*                         (1-ray.terminal.elements[1])*                         Jacobians.calcDetMapJacobian(ray.terminal.elements,
                                                      [Const.au,1,np.pi,2*np.pi,2*np.pi,2*np.pi])[0]
                    if ray.terminal.elements[1]<1:
                        ray.terminal.calcUelements(maxvalues=[Const.au,1,np.pi,2*np.pi,2*np.pi,2*np.pi])
                        x=ray.terminal.uelements[:5]
                        ph=mnd.pdf(x)
                    else:
                        ph=0
                except AssertionError as e:
                    detJ=0
                    ph=0
                detJs[i,j]+=np.abs(detJ)
                phs[i,j]+=ph
                pis[i,j]+=ph*np.abs(detJ)

fig=plt.figure(figsize=(6,6),constrained_layout=True)
ax=fig.gca()
ms=Map("sky")
ms.drawMap(ax)
ms.scatterMap(As,hs,marker='+',color='k')
cont=ms.area.contourf(As,hs,detJs,levels=1000,latlon=True,norm=LogNorm())
cbar=fig.colorbar(cont,drawedges=False,orientation="horizontal")
ms.scatterMap(Aloc*Angle.Rad,hloc*Angle.Rad,color='w',s=100,marker='v')

fig.savefig(f"figures/{pref}-sky-contdetJ.png")

fig=plt.figure(figsize=(6,6),constrained_layout=True)
ax=fig.gca()
ms=Map("sky")
ms.drawMap(ax)
ms.scatterMap(As,hs,marker='+',color='k')
ms.area.contour(As,hs,detJs,latlon=True,norm=LogNorm())
xs,ys=ms.area(As,hs)
cont=ax.contour(xs,ys,phs,levels=[0],colors=["w"])
cont=ax.contourf(xs,ys,phs,levels=100)
cbar=fig.colorbar(cont,drawedges=False,orientation="horizontal")
ms.scatterMap(Aloc*Angle.Rad,hloc*Angle.Rad,color='w',s=100,marker='v')

fig.savefig(f"figures/{pref}-sky-contpi.png")

fig=plt.figure(figsize=(6,6),constrained_layout=True)
ax=fig.gca()
ms=Map("sky")
ms.drawMap(ax)
ms.scatterMap(As,hs,marker='+',color='k')
xs,ys=ms.area(As,hs)
cont=ax.contour(xs,ys,phs,levels=[0],colors=["w"])
cont=ax.contourf(xs,ys,pis,levels=100) #,norm=LogNorm())
cbar=fig.colorbar(cont,drawedges=False,orientation="horizontal")
ms.scatterMap(Aloc*Angle.Rad,hloc*Angle.Rad,color='w',s=100,marker='v')

fig.savefig(f"figures/{pref}-sky-contph.png")

dt=elTime(1)

pis.sum(),pis.max()

phs.sum(),phs.max()

