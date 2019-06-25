#############################################################
#EXTERNAL LIBRARIES
#############################################################
#REGULAR
import spiceypy as spy
from commands import getoutput as System
from sys import argv,exit
from time import time
import numpy as np
from scipy.interpolate import interp1d
import hashlib

#GRAPHICAL
from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt,cm
from mpl_toolkits.mplot3d import Axes3D as plt3d,art3d
from mpl_toolkits.basemap import Basemap as Map,shiftgrid as Grid

#import visual as v

#DATABASE
import MySQLdb as mdb

#############################################################
#MACROS
#############################################################
norm=np.linalg.norm
rand=np.random.rand
PI=np.pi
DEG=PI/180
RAD=1/DEG
TAB="\t"

# Harvesine Formulae
HAV=lambda theta:np.sin(theta/2)**2
HAV2=lambda costheta:(1-costheta)/2
DIST=lambda p1,p2:norm(p1-p2)

#############################################################
#CONSTANTS
#############################################################
#TIME UNITS
SEC=1.0
MIN=60.0
HOUR=60.0*MIN
DAY=24*HOUR
YEAR=365.25*DAY

#UNITCONVERSION
UL=1.49597870691E8 #km
UV=2.97824659599673905e+01 #UL/UT

#FACTOR CONVERSION
STATECONV=np.array([UL,UL,UL,UV,UV,UV])

#OBJECTS
OBJECTS={"0":"Sun",
         "1":"Mercury",
         "2":"Venus",
         "3":"Earth",
         "4":"Moon",
         "5":"Mars",
         "6":"Jupiter",
         "7":"Saturn",
         "8":"Uranus",
         "9":"Neptune"}

###################################################
#DATABASE CONNECTION
###################################################
DATABASE="MinorBodies"
USER="minorbodies"
PASSWORD="347940"
CON=mdb.connect("localhost",USER,PASSWORD,DATABASE)
DB=CON.cursor()

#############################################################
#ROUTINES
#############################################################
def vec2str(vec,frm="%+.17e "):
    string=""
    for comp in vec:
        string+=frm%comp
    return string
    
def out2state(out,ini=0,end=-1):
    state=np.array([float(x) for x in out.split(" ")[ini:end]])
    return state

def getScenario(sfile):
    scenario=np.loadtxt(sfile)
    ncols=scenario.shape[1]
    nobjs=(ncols-1)/7
    obj=dict()
    obj["ts"]=scenario[:,0]
    for i in xrange(nobjs):
        stride=1+7*i
        dataobj=scenario[:,stride:stride+7]
        iobj=int(dataobj[0,0])
        obj["%d"%iobj]=dataobj[:,1:]

    return obj

def out2dict(output):
    parts=output.split("\n")
    comps=parts[0].split(",")
    i=0
    output=dict()
    for part in parts[1:]:
        if '(' in comps[i]:
            values=part.split(" ")
            output[comps[i]]=np.array([float(value) for value in values[:-1]])
        else:
            output[comps[i]]=float(part)
        i+=1
    return output
    
ETIME=time()
TIME=time()
def timeIt(msg='',stream=None):
    global TIME,ETIME
    ETIME=time()-ETIME
    total=time()-TIME
    msg="[Elapsed %s: last %.3f (%.0f msec), total %.3f sec]"%(msg,ETIME,1e3*ETIME,total)
    if stream is None:print msg
    else:print>>stream,msg
    ETIME=time()

class dict2obj(object):
    def __init__(self,dic={}):self.__dict__.update(dic)
    def __add__(self,other):
        for attr in other.__dict__.keys():
            exec("self.%s=other.%s"%(attr,attr))
        return self

def mysqlSelect(selection="*",table="Bodies",condition="limit 100",typereturn="dict"):
    """
    Executes a selection on a given table of the Database.

    Parameters:

       selection: which fields do you want to select from the table.  This is a comma separated list
                  of fields.  You should use ", " (comma+space) to separate fields.

       table: which table from the database.

       condition: condition for selection.  This is the part of the SQL command following the "from <TABLE>".

            Examples: where Name like '%zuluaga'
                      limit 1000
                      order by e
    """
    #QUERY
    sql="select %s from %s %s"%(selection,table,condition)
    DB.execute(sql)
    results=DB.fetchall()

    if selection=="*":
        #RECOVER ALL COLUMNS IN TABLE
        DB.execute("show columns from %s"%table)
        fields=DB.fetchall()
    else:
        #ONLY COLUMNS IN SELECTION
        fields=[(field,) for field in selection.split(", ")]

    dresults=[]
    for result in results:
        if typereturn=='dict':row=dict()
        else:row=[]
        for i in xrange(len(fields)):
            field=fields[i][0]
            if typereturn=='dict':row[field]=result[i]
            else:row+=[result[i]]
        dresults+=[row]

    return dresults
    
def listdict2matrix(listdict,keys=[]):
    n=len(listdict)
    m=len(listdict[0].keys())
    matrix=np.zeros((n,m))
    i=0
    for obj in listdict:
        if len(keys)==0:keys=obj.keys()
        matrix[i,:]=np.array([float(obj[key]) for key in keys])
        i+=1
    return matrix
    
def generateVelocities(velcumfile,nsample):
    velcum=np.loadtxt(velcumfile)
    ifvelcum=interp1d(velcum[:,1],velcum[:,0])
    v=ifvelcum(np.random.random(nsample))
    return v

def lat2str(lat):
    return "%g"%lat

def lon2str(lon):
    if lon>270:lon-=360
    return "%g"%lon

def drawMap(proj='robin',
            proj_opts=dict(resolution='c',lon_0=0),
            pars=np.arange(-60.,115,15.),
            pars_opts=dict(labels=[1,1,0,0],labelstyle="+/-",fontsize=8),
            mers=np.arange(-360.,360.,30.),
            mers_opts=dict(labels=[0,0,1,1],labelstyle="+/-",fontsize=8),
            coasts=False,
            fill=False,
            coasts_opts=dict(linewidth=0.5),
            fill_opts=dict(color='g',alpha=0.3,lake_color='aqua'),
            ):

    m=Map(projection=proj,**proj_opts)
    m.drawmapboundary()
    parallels=m.drawparallels(pars,**pars_opts)
    try:
        meridians=m.drawmeridians(mers,**mers_opts)
    except:pass
    if coasts:
        m.drawcoastlines(**coasts_opts)
    if fill:
        m.fillcontinents(**fill_opts)
    return m

def plotMap(map,alpha,delta,**args):
    x,y=map(alpha,delta)
    plt.plot(x,y,**args)

def normedArcDistance(p1,p2):
    """
    Compute arc distance for points in the unit circle
    
    Points are pairs [x,y]
    Such that 
       x=longitude/pi \in [-1,1]
       y=sin(latitude) \in [-1,1]
    """

    #Points
    x1,y1=p1
    x2,y2=p2
    
    #Derivative quantities
    c1=np.sqrt(1-y1**2) # cos(latitude)
    c2=np.sqrt(1-y2**2)
    cosdb=c1*c2+y1*y2 # cos(latitue2-latitude1)

    l1=np.pi*x1 # longitude
    l2=np.pi*x2

    #Haversine
    h=HAV2(cosdb)+c1*c2*HAV(l2-l1)
    
    #Angular distance
    delta=2*np.arcsin(np.sqrt(h))
    
    return delta

def arcDistance(s1,s2):
    """
    Compute arc distance for points in longitud, latitude

    Both, input angles and returned distance are in radians 
    """
    #Points
    l1,b1=s1
    l2,b2=s2
    
    #Haversine
    h=HAV(b2-b1)+np.cos(b1)*np.cos(b2)*HAV(l2-l1)

    #Angular distance
    delta=2*np.arcsin(np.sqrt(h))
    
    return delta

def car2sph(p):
    """
    Convert from unit square spherical coordinates to longitude latitude.

    p is a point in the unit square [[-1,1],[-1,1]]

    Return angles are in radians in [[-pi,pi],[-pi/2,pi/2]]
    """
    l=PI*p[0]
    b=np.arcsin(p[1])
    return np.array([l,b])

def sph2car(p):
    """
    Convert from unit longitude,latitude to unit square.

    p \in [[-pi,pi],[-pi/2,pi/2]]

    Return [x,y] \in [[-1,1],[-1,1]]
    """
    x=p[0]/PI
    y=np.sin(p[1])
    return np.array([x,y])

def s2d(g,m=0,s=0):
    return g+m/60.+s/3600.

def histOutline(histIn,binsIn):
    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)

def cumDistrib(x,h,x0=None,xn=None):
    norm=h.sum()
    F=np.zeros_like(h)
    F[0]=h[0]
    for i in xrange(1,h.shape[0]):
        F[i]=F[i-1]+h[i]
    F=F/(1.0*norm)
    if x0 is not None:
        x=np.concatenate(([x0],x))
        F=np.concatenate(([0],F))
    if xn is not None:
        x=np.concatenate((x,[xn]))
        F=np.concatenate((F,[1]))
    Fcum=np.vstack((x,F)).transpose()
    return Fcum

def MD5STR(str,len=-1):
    MD5=hashlib.md5()
    MD5.update(str)
    return MD5.hexdigest()[:len].upper()

#Weighting factor for computing density
def wFunction(d,h):
    """
    Schoenber B-spline function
    See: https://arxiv.org/pdf/1012.1885.pdf

    Plot:
        h=0.1
        sigma=wNormalization(h)
        fig=plt.figure()
        ax=fig.gca()
        ds=np.linspace(0,5*h,100)
        ws=np.array([sigma*wFunction(d,h) for d in ds])
        ax.plot(ds,ws)
        fig.savefig("scratch/weighting-shoenberg.png")

    Test it:
        from scipy.integrate import quad
        wnorm=lambda d:wFunction(d,h)*sigma
        print quad(wnorm,0,2*h)
    """
    q=d/h
    if q<1:w=0.25*(2-q)**3-(1-q)**3
    elif q<2:w=0.25*(2-q)**3
    else:w=0
    return w

def wNormalization(h):
    from scipy.integrate import quad
    sigma=1/quad(wFunction,0,2*h,args=(h,))[0]
    return sigma

def drummondDistance(q,e,i):
    """
    Drummond (1981)
    """
    #distform="POW(Perihelion_dist-%.17e,2)+POW(e-%.17e,2)+POW((i-%.17e)/45.,2)"%(q,e,i)
    distform="POW((Perihelion_dist-%.17e)/(Perihelion_dist+%.17e),2)+POW((e-%.17e)/(e+%.17e),2)+POW((i-%.17e)/180.,2)"%(q,q,e,e,i)
    return distform

def zappalaDistance(a,e,sini,O,o,num=0):
    """
    Zappala (1990), Nervorny & Vokrouhlicky (2006)
    am=(a+at)/2
    d2c=1/np.sqrt(am)*(ka*((at-a)/am)**2+ke*(et-e)**2+ki*(sinit-np.sin(i*DEG))**2+kO*(Omega-Ot)**2+kw*(omega-ot)**2)
    """
    ka=5./4
    ke=ki=2

    #ORIGINAL ZAPPALA
#     kw=kO=1e-4
#     distform="""\
# 1/SQRT((a+%.17e)/2)*(\
# %.17e*POW((a-%.17e)/((a+%.17e)/2),2)+\
# %.17e*POW(e-%.17e,2)+\
# %.17e*POW(sini-%.17e,2)+\
# %.17e*POW(Node-%.17e,2)+\
# %.17e*POW(Peri-%.17e,2)\
# )"""%(a,ka,a,a,ke,e,ki,sini,kO,O,kw,o)

    #MODIFIED ZAPPALA
    kw=kO=1e-4
    distform="""\
1/SQRT((a+%.17e)/2)*(\
%.17e*POW((a-%.17e)/((a+%.17e)/2),2)+\
%.17e*POW(e-%.17e,2)+\
%.17e*POW(sini-%.17e,2)+\
%.17e*POW(ABS((Node-%.17e+180)MOD(360)-180),2)+\
%.17e*POW(ABS((Peri-%.17e+180)MOD(360)-180),2)\
)"""%(a,ka,a,a,ke,e,ki,sini,kO,O,kw,o)

    return distform

def theoFlux_DoubleTrigCos(q,f,a,b):
    if q<=90:
        fv=f*np.cos(np.pi/2-q*DEG)**a
    else:
        fv=f*np.cos(np.pi/2-q*DEG)**b
    return fv
theoFlux_DoubleTrigCos=np.vectorize(theoFlux_DoubleTrigCos)

def grtID(date,dirfile,qvel,velfile,name):

    # DATE ANALYSIS
    parts=date.split(" ")
    dparts=parts[0].split("/")
    tparts=parts[1].split(":")
    tstring="%04d%02d%02d%02d%02d%02d"%(int(dparts[2]),int(dparts[0]),int(dparts[1]),
                                        int(tparts[0]),int(tparts[1]),int(tparts[2]))
    out=System("./whattimeisit.exe '%s' ET > /dev/null"%date)
    t=float(out.split("\n")[4])
    
    # MAKE STRING
    dirmd5=System("md5sum %s |awk '{print $1}'"%dirfile)
    velmd5=System("md5sum %s |awk '{print $1}'"%velfile)
    makestr="qvel=%d & name=%s & dirmd5=%s & velmd5=%s"%(qvel,name,dirmd5,velmd5)
    
    md5str=MD5STR(makestr,len=6)
    
    # RUN STRING
    runstr="%s-%s"%(tstring,md5str)

    return runstr
