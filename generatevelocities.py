from gravray import *

###################################################
#USAGE
###################################################
usage="""
Generate a list of velocities from a given probability source.

python generatevelocities.py [<nsample>] [<source>] [<method>]

Where:

  nsample: number of velocities to be generated.

  source: which source of objects velocities do you want to use.
          Available sources are: velimp (impact velocities), velinf
          (velocitt at infinite), velast (velocities with respect to
          SSB)

  method: generation method, either random (following PDF) or regular
          (spaced proportionally to probability)

"""

###################################################
#INPUTS
###################################################
iarg=1
try:
    nsample=int(argv[iarg])
    iarg+=1
except:nsample=10
try:
    source=argv[iarg]
    iarg+=1
except:source="velimp"

try:
    method=argv[iarg]
    iarg+=1
except:method="regular"

print "Generating %d velocities from source '%s' with distribution '%s'..."%(nsample,source,method)

###################################################
#LOAD REQUIRED INFORMATION
###################################################
velfile="velocities-%s.dat"%(source)
vp=np.loadtxt("util/data/%s-pdf.data"%source)
vmin=vp[:,0].min();vmax=vp[:,0].max();

###################################################
#GENERATE VELOCITIES
###################################################
#RANDOM 
if method=="random":
    vs=generateVelocities("util/data/%s-cum.data"%source,nsample)

#UNIFORMLY DISTRIBUTED IN CUMMULATIVE 
else:
    du=1./nsample
    u=np.linspace(du,1,nsample)
    velcum=np.loadtxt("util/data/%s-cum.data"%source)
    ifvelcum=interp1d(velcum[:,1],velcum[:,0])
    vs=ifvelcum(u)

###################################################
#SAVE VELOCITIES
###################################################
print "Saving sample with %d velocities in '%s'..."%(nsample,velfile)
np.savetxt(velfile,vs)

###################################################
#HISTOGRAM
###################################################
nbins=vp.shape[0]
h,x=np.histogram(vs,nbins,(vmin,vmax))
fig=plt.figure()
ax=fig.gca()
ax.plot(x[:-1],h,'-')
ax.plot(vs,np.zeros_like(vs),'k+',ms=10)
ax.plot(vp[:,0],vp[:,1]*nsample,'-')
fig.savefig("scratch/%s-pdf-sample.png"%source)
