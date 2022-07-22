import numpy as np
import math
import time
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits
from odlib import *

# angles/radians (declination and right ascension)

def decToRad(dec:float)->float:
    """ Given decimal value return in radians
        Args:
            dec (float): decimal value
        Returns:
            float: value in radians
    """
    return float(dec)/180*math.pi

def HMStoDeg(h:float,m:float,s:float,convert=False)->float:
    """ Given HMS, return in degrees or radians
        Args:
            h (float): hours
            m (float): minutes
            s (float): seconds
            convert (bool): True to convert to radians
        Returns:
            float: value in degrees or radians
    """
    try: h,m,s=float(h),float(m),float(s)
    except: raise Exception("Values must be floats")
    res = h*15 + m/60*15 + s/3600*15
    res%=360
    return np.deg2rad(res) if convert else res

def DMStoDeg(d:float,arcm:float,arcs:float,convert=False)->float:
    """ Given DMS, return in degrees or radians
        Args:
            d (float): degrees
            m (float): minutes
            s (float): seconds
            convert (bool): True to convert to radians
        Returns:
            float: value in degrees or radians
    """
    return np.deg2rad(math.copysign(abs(d)+arcm/60+arcs/3600,d)) if convert else math.copysign(abs(d)+arcm/60+arcs/3600,d)

def RAdecimalToHMS(dec:float)->tuple:
    """ Converts from decimal to HMS (right ascension)
        Args:
            dec (float): decimal value
        Returns:
            tuple: (hours, minutes, seconds)
    """
    dec%=360
    h=dec/15
    m=(dec%15)/15*60
    s=m%1*60
    h//=1
    m//=1
    return (h,m,s)

def DECdecimalToDMS(dec:float)->tuple:
    """ Converts from decimal to DMS (declination)
        Args:
            dec (float): decimal value
        Returns:
            tuple: (degrees, minutes, seconds)
    """
    d=math.trunc(dec)
    dec=abs(dec)-abs(d)
    m=math.trunc(dec*60)
    dec-=m/60
    s=dec*3600
    return (d,m,s)

def getAngle(sin:float,cos:float)->float:
    """ Returns the angle (in radians) in the correct quadrant given sin and cos
        Args:
            sin (float): sin value
            cos (float): cos value
        Returns:
            float: resulting angle in radians
    """
    return math.atan2(sin,cos) % (math.pi*2)

# vector stuff

def cross(v1:list, v2:list)->list:
    """ Returns the cross product of two 3D vectors
        Args:
            v1 (list): vector1
            v2 (list): vector2
        Returns:
            list: resulting cross product
    """
    return [(v1[1]*v2[2] - v1[2]*v2[1]),-(v1[0]*v2[2] - v1[2]*v2[0]),(v1[0]*v2[1] - v1[1]*v2[0])]

def dot(v1:list,v2:list)->float:
    """ Returns the dot product of two vectors
        Args:
            v1 (list): vector1
            v2 (list): vector2
        Returns:
            float: resulting dot product
    """
    return sum([v1[i]*v2[i] for i in range(len(v1))])

def det2x2(m)->float:
    """ Returns the determinant of a 2x2 matrix
        Args:
            m (list): matrix
        Returns:
            float: resulting determinant
    """
    return m[0][0]*m[1][1]-m[0][1]*m[1][0]

def getMag(vec:list)->float:
    """ Returns the magnitude given a vector
        Args:
            vec (list): vector
        Returns:
            float: the magnitude of the vector
    """
    return math.sqrt(sum([vec[i]**2 for i in range(len(vec))]))
    
def rotZX(v:list,alpha:float,beta:float)->list:
    """ Rotates a vector around z axis with alpha, then rotates around x axis with beta'''
        Args:
            v (list): original vector
            alpha (float): angle to rotate around z axis (in radians)
            beta (float): angle to rotate around x axis (in radians)
        Returns:
            list: rotated vector
    """
    z=np.array([[np.cos(alpha),-np.sin(alpha),0],
                [np.sin(alpha),np.cos(alpha),0],
                [0,0,1]])
    x=np.array([[1,0,0],
                [0,np.cos(beta),-np.sin(beta)],
                [0,np.sin(beta),np.cos(beta)]])
    return np.dot(np.matmul(x,z),v)

def rotX(v:list,omega:float)->list:
    """ Rotates a vector around x axis with omega (in radians)'''
        Args:
            v (list): original vector
            omega (float): angle to rotate around x axis (in radians)
        Returns:
            list: rotated vector
    """
    x=np.array([[1,0,0],
                [0,np.cos(omega),-np.sin(omega)],
                [0,np.sin(omega),np.cos(omega)]])
    return np.matmul(x,v)

# assorted other functions

def newton(func,der,init:float,err:float)->float: #function, derivative, inital_guess, error_tolerance
    """ Performs the Newton-Rhapson method
        Args:
            func (function): function
            der (function): derivative of function
            init (float): initial guess
            err (float): tolerance
        Returns:
            float: the answer
    """
    prev=-1e10
    while abs(init-prev)>=err:
        prev,init=init,init-func(init)/der(init)
    return init

def error(acc:float,exp:float)->float:
    """ Returns the error given accepted and experimental values
        Args:
            acc (float): accepted value
            exp (float): experimental value
        Returns:
            float: the error in percent
    """
    return math.copysign(abs(acc-exp)/acc*100,1)


# OD code
class ODElements:
    '''Class that represents and calculates orbital elements'''
    
    def __init__(self, pos:list, vel:list, time:float):
        """ Initializes ODElements class
            Args:
                pos (list): position of asteroid
                vel (list): velocity of asteroid (non Gaussian)
                time (float): given time in julian days
            Returns:
                None
        """
        # constants
        self.k = 0.0172020989484 # au^(3/2)/day
        
        self.time=time
        self.vel=vel
        self.vel=np.array([self.vel[i]/(2*math.pi)*365.2568983 for i in range(3)]) # convert to Gaussian
        self.pos=pos
        self.angMoment=self.getAngMoment()
        
        self.mu=1
        self.a=self.getSemiMajor() # semi-major axis
        self.e= self.getEcc() # eccentricity
        self.i=self.getInc() # inclination
        self.o=self.getLongAsc() # longitude of ascending node
        self.v=self.getTrueAnom() # true anomaly
        self.w=self.getArgPer() # argument of perihelion
        self.M=self.getMeanAnomaly() # mean anomaly
        self.T=self.getPeriT() # time of perihelion passage T
      
    def getInfo(self)->list:
        """ Returns info from given day from initialization
            Args:
                None
            Returns:
                list: all info
        """
        return self.info
    
    def getPos(self)->list:
        """ Returns the position of the asteroid
            Args:
                None
            Returns:
                list: the position
        """
        return self.pos
    
    def getVel(self)->list:
        """ Returns the velocity of the asteroid
            Args:
                None
            Returns:
                list: the velocity
        """
        return self.vel
    
    def getPosMag(self)->float:
        """ Returns the magnitude of the position vector
            Args:
                None
            Returns:
                float: magnitude of position
        """
        return math.sqrt(sum([self.pos[i]**2 for i in range(3)]))
    
    def getVelMag(self)->float:
        """ Returns the magnitude of the velocity vector
            Args:
                None
            Returns:
                float: magnitude of velocity
        """
        return math.sqrt(sum([self.vel[i]**2 for i in range(3)]))
    
    def getAngMoment(self)->list:
        """ Calculates and returns the specific angular momentum
            Args:
                None
            Returns:
                list: specific angular momentum components
        """
        pos,vel=self.getPos(),self.getVel()
        res=cross(pos,vel)
        return np.array([res[0], res[1], res[2]])
    
    def getAngMomentMag(self)->float:
        """ Returns the magnitude of the specific angular momentum
            Args:
                None
            Returns:
                float: the magnitude of specific angular momentum
        """
        return math.sqrt(sum([self.angMoment[i]**2 for i in range(3)]))
    
    def getSemiMajor(self)->float:
        """ Calculates and returns the semi major axis using vis-viva
            Args:
                None
            Returns:
                float: the semi major axis
        """
        return 1/(2/self.getPosMag() - dot(self.vel,self.vel)/self.mu)
    
    def getEcc(self)->float:
        """ Calculates and returns the eccentricity
            Args:
                None
            Returns:
                float: eccentricity
        """
        return math.sqrt(1-dot(self.angMoment, self.angMoment)/(self.mu*self.a))
    
    def getInc(self, rad:bool=False)->float:
        """ Calculates and returns the inclination in degrees
            Args:
                rad (bool): True if return in radians
            Returns:
                float: the inclination in degrees or radians
        """
        return math.acos(self.angMoment[2]/self.getAngMomentMag()) if rad else np.rad2deg(math.acos(self.angMoment[2]/self.getAngMomentMag()))
    
    def getLongAsc(self, rad:bool=False):
        """ Calculates and returns the longitude of ascending node in degrees
            Args:
                rad (bool): True if return in radians
            Returns:
                float: the longitude of ascending node in degrees or radians
        """
        s=self.angMoment[0]/(self.getAngMomentMag()*math.sin(np.deg2rad(self.i)))
        c=-self.angMoment[1]/(self.getAngMomentMag()*math.sin(np.deg2rad(self.i)))
        return getAngle(s,c) if rad else np.rad2deg(getAngle(s,c))
    
    def getArgPer(self, rad:bool=False)->float:
        """ Calculates and returns the argument of perihelion in degrees
            Args:
                rad (bool): True if return in radians
            Returns:
                float: the longitude of ascending node in degrees or radians
        """
        x,y,z=self.pos[0],self.pos[1],self.pos[2]
        s=z/(self.getPosMag()*math.sin(np.deg2rad(self.i)))
        c=(x*math.cos(np.deg2rad(self.o)) + y*math.sin(np.deg2rad(self.o)))/self.getPosMag()
        U=np.rad2deg(getAngle(s,c))
        return np.deg2rad((U-self.v)%360) if rad else (U-self.v)%360
    
    def getTrueAnom(self, rad:bool=False)->float:
        """ Calculates and returns the true anomaly
            Args:
                rad (bool): True if return in radians
            Returns:
                float: the true anomaly in degrees or radians
        """
        c=1/self.e*(self.a*(1-self.e**2)/self.getPosMag() - 1)
        s=self.a*(1-self.e**2)/(self.getAngMomentMag() * self.e)*dot(self.pos,self.vel)/self.getPosMag()
        return getAngle(s,c) if rad else np.rad2deg(getAngle(s,c))
    
    def getPeriT(self)->float:
        """ Calculates and returns the time of perihelion
            Args:
                None
            Returns:
                float: the time of perihelion
        """
        n=self.k/(self.a**(3/2))
        return self.time-np.deg2rad(self.M)/n
    
    def getMeanAnomaly(self,rad:bool=False)->float:
        """ Calculates and returns the mean anomaly
            Args:
                rad (bool): True if return in radians
            Returns:
                float: the mean anomaly in degrees or radians
        """
        s=(self.getPosMag()*math.sin(np.deg2rad(self.v)))/(self.a*np.sqrt(1-self.e**2))
        c=(self.e+np.cos(np.deg2rad(self.v)))/(1+self.e*np.cos(np.deg2rad(self.v)))
        E=getAngle(s,c)
        return E-self.e*np.sin(E) if rad else np.rad2deg(E-self.e*np.sin(E))
    
    def getTimeMeanAnomaly(self, time:float, date:str)->float:
        """ Calculates mean anomaly for given date
            Args:
                time (float): time in Julian days for the Mean Anomaly
                date (str): date for Mean Anomaly
            Returns:
                float: mean anomaly in degrees
        """
        n=self.k*np.sqrt(self.mu/(self.a**3))
        M=np.rad2deg(n*(time-self.T))
        return M
    
    def printError(self, results:list):
        """ Prints everything
            Args:
                None
            Returns:
                None
        """
        # EC, QR, IN, OM, W, Tp, N, MA, TA, A, AD, PR,
        print("Semi-major axis:", "\n\tactual:",results[9], "\n\tcalculated:",self.a, "\n\terror:",error(results[9],self.a))
        print("Eccentricity:", "\n\tactual:",results[0], "\n\tcalculated:",self.e, "\n\terror:",error(results[0],self.e))
        print("Inclination:","\n\tactual:",results[2],"\n\tcalculated:",self.i, "\n\terror:",error(results[2],self.i))
        print("Longitude of Ascending Node:","\n\tactual:",results[3],"\n\tcalculated:",self.o, "\n\terror:",error(results[3],self.o))
        #print("True anomaly:",results[8],self.v,error(results[8],self.v))
        print("Argument of perihelion:","\n\tactual:",results[4],"\n\tcalculated:",self.w, "\n\terror:",error(results[4],self.w))
        print("Time of Perihelion Passage T:","\n\tactual:",results[5],"\n\tcalculated:", self.T,"\n\terror:",error(results[5],self.T))
        #print("Mean Anomaly:",results[7],self.M,error(results[7],self.M))
        #print(od.a, od.e, od.i, od.o, od.v, od.w)
        
    def getElements(self):
        """ Returns all orbital elements
            Args:
                rad (bool): True if return in radians
            Returns:
                floats: a,e,i,o,v,w,T,M
        """
        return self.a, self.e, self.i, self.o, self.v, self.w, self.T, self.M

# Data class
class Data:
    '''Class that reads and interprets data from input file'''
    
    def __init__(self):
        """ Initializes the class
            Args:
                None
            Returns:
                None
        """ 
        self.info=np.array([]) # non formatted data from input file
        self.inputData=None # formatted data from input file (list)]
        
        self.infoByTime={}
        self.infoByDate={} # dictionary of values. date is key. format for key: 2018-Jul-14 00:00:00.0000
        self.sunFileName=""
        self.inputFileName=""
        
        # constants
        self.JDTIME=0
        self.DATE=1
        self.RA=2
        self.DEC=3
        self.R=4
        
    def getInput(self, file:str)->list:
        """ Formats and returns formatted input. Stores into dictionaries, and converts all RA and Dec 
            into decimals. 
            Args:
                file (str): the input file name
            Returns:
                list: the formatted input [jdtime (float), date (str), ra (float), dec (float), [RX,RY,RZ] (np.array)]
        """ 
        self.inputFileName=file
        self.info=np.loadtxt(file,dtype=str,delimiter=",")
        # store in dictionary for fast retrieval; also, formats all the dec and ra, converting to decimals
        # [jdtime, date(str), ra, dec, [RX,RY,RZ] (np.array)]
        self.inputData=[]

        for line in range(1,len(self.info)):
            data=self.info[line]
            jdtime = float(data[0])
            date = data[1].strip()
            
            strRA,strDec = data[2].split(':'), data[3].split(':')
            h,m,s=float(strRA[0]),float(strRA[1]),float(strRA[2])
            ra=HMStoDeg(h,m,s)
            d,m,s=float(strDec[0]),float(strDec[1]),float(strDec[2])
            dec=DMStoDeg(d,m,s)
            
            R=[float(data[4]),float(data[5]),float(data[6])]
            
            self.inputData.append([jdtime, date, ra, dec, R])
            self.infoByDate[date] = [jdtime, date, ra, dec, R]
            self.infoByTime[jdtime] = [jdtime, date, ra, dec, R]
            
        return self.inputData # nothing is formatted, just all information
        
    def getSunInput(self,date:str=None,time:float=None)->list:
        """ Returns the sun input for a given time
            Args:
                date (str): optional; the date for sun vector
                time (float): optional; the time in Julian days for sun vector
            Returns:
                list: the sun vector
        """ 
        if np.shape(self.info)==0: raise Exception("No input has been loaded up")
        if not(date==None):
            return self.infoByDate[date][self.R]
            
        elif not(time==None):
            return self.infoByTime[time][self.R]
        
        else: # nothing has been inputted, throw an exception
            raise Exception("No time has been given to find sun input")
        
    def getRADECInput(self,date:str=None,time:float=None):
        """ Returns the right ascension and declination for a given time
            Args:
                date (str): optional; the date for right ascension and declination
                time (float): optional; the time in Julian days for right ascension and declination
            Returns:
                floats: ra, dec
        """ 
        if np.shape(self.info)==0: raise Exception("No input has been loaded up")
        if not(date==None):
            d=self.infoByDate[date]
            return d[self.RA], d[self.DEC]
            
        elif not(time==None):
            d=self.infoByTime[time]
            return d[self.RA], d[self.DEC]
            
        else: # nothing has been inputted, throw an exception
            raise Exception("No time has been given to find ra")
            
    def getJDTime(self,date:str)->float:
        """ Returns Julian Days time given date
            Args:
                date (str): optional; the date for right ascension and declination
                time (float): optional; the time in Julian days for right ascension and declination
            Returns:
                floats: ra, dec
        """ 
        if np.shape(self.info)==0: raise Exception("No input has been loaded up")
        
        d=self.infoByDate[date]
        return d[self.JDTIME]

    
    def formatTestInputInfo(self, file:str):
        """ Returns re-formatted data from test input file (for testing purposes, specifically OD elements generation)
            Args:
                file (str): file name
            Returns:
                lists: time (in julian days), data [[x,y,z],[dx,dy,dz]], timestamps (strings)
        """
        info=np.loadtxt(file,dtype=str,delimiter=",")
        time=np.array([float(info[i,0]) for i in range(1,len(info))])
        timestamps=np.copy(info[1:,1])
        
        return time,np.array([([float(info[i][2]),float(info[i][3]),float(info[i][4])], 
                        [float(info[i][5]),float(info[i][6]),float(info[i][7])]) for i in range(1,len(info))]), timestamps
   
    def getTestInput(self, file:str, date:str):
        """ Returns pos, vel, and times for testing asteroid (for testing purposes, specifically OD elements generation)
            Args:
                file (str): file name
                date (str): date for testing
            Returns:
                lists: pos, vel, time
        """
        times,data,timestamps=self.formatTestInputInfo(file)           
        line = 0
        for i in range(len(timestamps)):
            if date in timestamps[i]: 
                line = i
                break
                
        time,info=times[line],data[line]
        pos,vel=info[0],info[1]
        return pos, vel, time
                               
    
    def getSunPos(self, date:str, file:str)->list:
        """ Gets the vector from the Earth to the Sun given the date
            Args:
                date (str): the date to use
                file (str): file name
            Returns:
                list: sun vector R
        """
        info=np.loadtxt(file,dtype=str,delimiter=",")
        
        timestamps=info[:,1]
        flag=False
        for i in range(len(timestamps)):
            if date in timestamps[i]: 
                line = i
                flag=True
                break
        
        if not flag: return np.array([0,0,0]) # not found
        
        stuff=info[line,:]
        x,y,z=float(stuff[2]),float(stuff[3]),float(stuff[4])
        return np.array([x,y,z])
    
    def getAllSunPos(self, file:str)->list:
        """ Gets all sun positions and returns as a list
            Args:
                file (str): file name
            Returns:
                list: list of lists [time, date, sunPos (np.array)]
        """
        info=np.loadtxt(file,dtype=str,delimiter=",")
        results=[]
        
        for line in info[1:]:
            time,date=float(line[0]),line[1].strip()
            R=np.array([float(line[2]),float(line[3]),float(line[4])])
            results.append([time,date,R])
  
        return results

    def exportEphemeris(self, fileName:str, results:list, actualEph:str):
        """ Exports ephemeris to a file
            Args:
                fileName (str): file name for exported ephemeris
                results (list): all results
                actualEph (str): file path for actual ephemeris from JPL horizons
            Returns:
                None
        """
        actualRA,actualDec=[],[]
        if not(actualEph==""):
            data=np.loadtxt(actualEph,dtype=str,delimiter=",")[1:,3:]
            for i in range(len(data)):
                ra=data[i][0].split()
                dec=data[i][1].split()
                actualRA.append(HMStoDeg(float(ra[0]),float(ra[1]),float(ra[2])))
                actualDec.append(DMStoDeg(float(dec[0]),float(dec[1]),float(dec[2])))
        with open(fileName, 'w') as file:
            file.write(("Date\tTime\tRA\tDec\tRA error\tDec error\n").expandtabs(35))
            counter=0
            for date, time, ra, dec in results:
                err=""
                if not (actualEph==""):
                    cra=ra.split()
                    cdec=dec.split()
                    err="\t"
                    err+=str(error(actualRA[counter],HMStoDeg(float(cra[0]),float(cra[1]),float(cra[2]))))
                    err+="\t"
                    err+=str(error(actualDec[counter],DMStoDeg(float(cdec[0]),float(cdec[1]),float(cdec[2]))))
                file.write((date+"\t"+time+"\t"+ra+"\t"+dec+err+"\n").expandtabs(35))
                counter+=1
                
    def exportMonteCarlo(self, fileName:str, vals:list, results):
        """ Exports Monte Carlo to a file
            Args:
                fileName (str): file name for exported ephemeris
                vals (list): all vals [num,a,e,i,o,w,T]
                results (list): real values to compare vals [a,e,i,o,w,t]
            Returns:
                None
        """
        labels=["a","e","i","o","w","T"]
        with open(fileName, 'w') as file:
            for j in range(len(vals)):
                file.write("----"+str(vals[j][0])+"----\n")
                for i in range(6):
                    file.write(labels[i]+": "+str(vals[j][i+1])+" \t error: " + str(error(results[i],vals[j][i+1])) + "\n")

    
    def printResults(self, fileName:str, pos:list, vel:list, rho:list, a:float, e:float, i:float, o:float, T:float, w:float, date:str, M:float):
        """ Gets the vector from the Earth to the Sun given the date
            Args:
                fileName (str): exported file name
                pos (list): position from sun
                vel (list): the velocity of the asteroid
                rho (list): the position from Earth
                a (float): semi major axis
                e (float): eccentricity
                i (float): inclination
                o (float): longitude of ascending node (Degrees)
                T (float): time of perihelion passage
                w (float): argument of perihelion (Degrees)
                date (str): date for calculated mean anomaly
                M (float): mean anomaly
            Returns:
                None
        """
        with open(fileName, 'w') as file:
            file.write("1999 GJ2 Orbit Determination")
            file.write("\n----------------------------------------------\n")
            file.write("Position From Sun (r, AU):\n\t" + str(pos))
            file.write("\nVelocity (AU/day):\n\t"+str(vel))
            file.write("\nPosition From Earth (rho, AU):\n\t"+str(rho))
            file.write("\n----------------------------------------------\n")
            file.write("Orbital Elements:")
            file.write("\n\tSemi-Major Axis (AU): " + str(a))
            file.write("\n\tEccentricity: " + str(e))
            file.write("\n\tInclination (deg): " + str(i))
            file.write("\n\tLongitude of Ascending Node (deg): " + str(o))
            file.write("\n\tTime of Perihelion Passage (Julian days): " + str(T))
            file.write("\n\tArgument of Perihelion (deg): " + str(w))
            file.write("\n----------------------------------------------\n")
            file.write("Mean Anomaly (deg) for "+date+":")
            file.write("\n\t"+str(M))
            
        

    
# Final OD Class
class OD:
    '''Class that performs all orbital determination calculations'''
    
    def __init__(self, inputFile:str):
        """ Initializes OD class
            Args:
                inputFile (str): input file name
            Returns:
                None
        """
        # constants
        self.k = 0.0172020989484 #Gaussian gravitational constant  
        self.cAU = 173.144643267 #speed of light in au/(mean solar)day  
        self.mu = 1
        self.eps = np.radians(23.4374) #Earth's obliquity
        self.toGaussian=365.2568983
        self.mu = 1
        
        self.data=Data() # Data object
        self.inputFile=inputFile
        
    def genElements(self, pos:list, vel:list, time:float, update:bool=True):
        """ Calculates and returns the orbital elements given position, velocity, time
            Args:
                pos (list): the position vector
                vel (list): the velocity vector
                time (float): the time in Julian days
                update (bool): if True keeps the newly calculated orbital elements
            Returns:
                floats: the orbital elements; a,e,i,o,v,w,T,M
        """
        if update:
            self.pos,self.vel,self.time=pos,vel,time
            self.od=ODElements(self.pos,self.vel,self.time)
            self.a,self.e,self.i,self.o,self.v,self.w,self.T,self.M = self.od.getElements()
            return self.getElements()
        else:
            od=ODElements(pos,vel,time)
            return od.getElements()
    
    def getElements(self):
        """ Returns the orbital elements (already calculated)
            Args:
                rad (bool): True if return in radians
            Returns:
                floats: a,e,i,o,v,w,T,M
        """
        return self.a, self.e, self.i, self.o, self.v, self.w, self.T, self.M
    
    def SEL(self, taus:list, sunR2:list, rhohat2:list, coefD:list):
        """ Scalar Equation of Lagrange to calculate the roots (r) and rhos corresponding to each r
            Args:
                taus (list): a list of floats of taus [T1,T3,T]
                sunR2 (list): a list representing the sun vector R2
                rhohat2 (list): a list representing the rhohat2 vector
                coefD (list): a list of D coefficients [D0,D21,D22,D23]
            Returns:
                lists: roots (r's), rhos
        """
        T1,T3,T=taus[0],taus[1],taus[2]
        D0,D21,D22,D23=coefD[0],coefD[1],coefD[2],coefD[3]
        A1=T3/T
        B1=A1/6*(T**2-T3**2)
        A3=-T1/T
        B3=A3/6*(T**2-T1**2)
        A=(A1*D21-D22+A3*D23)/(-D0)
        B=(B1*D21+B3*D23)/(-D0)
        
        E=-2*(dot(rhohat2, sunR2))
        F=dot(sunR2,sunR2)
        
        a=-(A**2+A*E+F)
        b=-self.mu*(2*A*B+B*E)
        c=-self.mu**2*B**2
        
        coef=[c,0,0,b,0,0,a,0,1]
        res=poly.polyroots(coef)

        temproots=[]
        for val in res: 
            if np.isreal(val) and np.real(val)>0: temproots.append(np.real(val))

        temprhos=[A+B/temproots[i]**3 for i in range(len(temproots))]
        
        # ignore pairs where rho magnitude is negative
        roots=[]
        rhos=[]
        #print(temproots, temprhos)
        for i in range(len(temproots)):
            if temprhos[i]>0.0:
                roots.append(temproots[i])
                rhos.append(temprhos[i])
                
        
        
        return roots,rhos
    
    def ephemeris(self, time:float, date:str, file:str="", sunPos:list=np.array([])):
        """ Calculates RA and Dec given time and date, using previously calculated orbital elements
            Args:
                time (float): time to determine ephemeris for in Julian Days
                date (str): date for which to determine ephemeris
                file (str): file name for sun positions; optional
            Returns:
                floats: ra, dec
        """
        n=self.k*math.sqrt(self.mu/(self.a**3))
        M=n*(time-self.T)
        E=newton(lambda E:M - E + self.e*np.sin(E), lambda E: -1+self.e*np.cos(E), M, 1e-17)
        
        pos=np.array([self.a*math.cos(E)-self.a*self.e, self.a*math.sqrt(1-self.e**2)*math.sin(E), 0])
        
        # the four rotations
        pos=rotZX(pos,np.deg2rad(self.w),np.deg2rad(self.i))
        pos=rotZX(pos,np.deg2rad(self.o),self.eps)
        
        if file=="": # no file given, use sunPos
            R=sunPos
        else: R=self.data.getSunPos(date, file)
        if np.array_equal(R, np.array([0,0,0])): raise Exception("Sun Position Not Found in SunPos.txt")
        rho=pos+R
        rhohat=rho/getMag(rho)
        
        dec=math.asin(rhohat[2])
        cra=rhohat[0]/math.cos(dec)
        sra=rhohat[1]/math.cos(dec)

        ra=getAngle(sra,cra)
        
        dec=np.rad2deg(dec)
        ra=np.rad2deg(ra)
        
        dec=DECdecimalToDMS(dec)
        ra=RAdecimalToHMS(ra)
        
        return ra,dec
        
        
    def fg(self, tau:float, r2mag:float, r2dot:list, order:int, r2:list=[]):
        """ Gets the f and g values given one time
            Args:
                tau (float): the time in Julian Days
                r2mag (float): the magnitude of r2
                r2dot (list): the velocity vector 2
                order (int): order of f and g taylor series approximations
                r2 (list): optional parameter, the position vector 2
            Returns:
                floats: f, g
        """
        if len(r2)==0: u=self.mu/r2mag**3
        else: u=self.mu/getMag(r2)**3
        
        f=1-1/2*u*tau**2
        g=tau
        
        if order>=3:
            z=dot(r2,r2dot)/(dot(r2,r2))
            q=dot(r2dot,r2dot)/(dot(r2,r2))-u
            f+=1/2*u*z*tau**3
            g+=-1/6*u*tau**3
        
        if order>=4:
            f+=1/24*(3*u*q-15*u*z**2+u**2)*tau**4
            g+=1/4*u*z*tau**4
        
        return f, g
        
    def getFGVals(self, tau1:float, tau3:float, r2mag:float, r2dot:list, order1:int, order2:int, r2:list=[]):
        """ Gets all f and g values
            Args:
                tau1 (float): the time in Julian Days for observation 1 from obs 2(T1-T2)
                tau3 (float): the time in Julian days for observation 3 from obs 2(T3-T2)
                r2mag (float): the magnitude of r2
                r2dot (list): the velocity vector 2
                order1 (int): Order of Taylor Series expansion for f and g values for observation 1
                order2 (int): Order of Taylor Series expansion for f and g values for observation 3
                r2 (list): optional parameter, the position vector 2
            Returns:
                lists: [f1,f3], [g1,g3]
        """
        f1,g1=self.fg(tau1,r2mag,r2dot,order1,r2)
        f3,g3=self.fg(tau3,r2mag,r2dot,order2,r2)
        return [f1,f3], [g1,g3]
    
    def getDCoef(self, ra:list, dec:list, R1:list, R2:list, R3:list):
        """ Gets the D coefficients given the ra and dec for three observations (in radians)
            Args:
                ra (list): the right ascensions for three observations (radians)
                dec (list): the declinations for three observations (radians)
                R1 (list): the sun vector for observation 1
                R2 (list): the sun vector for observation 2
                R3 (list): the sun vector for observation 3
            Returns:
                list: [D0,D11,D12,D13,D21,D22,D23,D31,D32,D33], [rhohat1, rhohat2, rhohat3]
        """
        ra1,ra2,ra3=ra[0],ra[1],ra[2]
        dec1,dec2,dec3=dec[0],dec[1],dec[2]
        
        rhohat1=np.array([np.cos(ra1)*np.cos(dec1), np.sin(ra1)*np.cos(dec1), np.sin(dec1)])
        rhohat2=np.array([np.cos(ra2)*np.cos(dec2), np.sin(ra2)*np.cos(dec2), np.sin(dec2)])
        rhohat3=np.array([np.cos(ra3)*np.cos(dec3), np.sin(ra3)*np.cos(dec3), np.sin(dec3)])
        
        D0=dot(rhohat1, cross(rhohat2,rhohat3))
        D11=dot(cross(R1, rhohat2),rhohat3)
        D12=dot(cross(R2, rhohat2),rhohat3)
        D13=dot(cross(R3, rhohat2),rhohat3)
        
        D21=dot(cross(rhohat1,R1), rhohat3)
        D22=dot(cross(rhohat1,R2), rhohat3)
        D23=dot(cross(rhohat1,R3), rhohat3)
        
        D31=dot(rhohat1, cross(rhohat2,R1))
        D32=dot(rhohat1, cross(rhohat2,R2))
        D33=dot(rhohat1, cross(rhohat2,R3))
     
        return [D0,D11,D12,D13,D21,D22,D23,D31,D32,D33], np.array([rhohat1, rhohat2, rhohat3])
    
    def MoGGenData(self,loaded:bool,selTime:list=[],selDate:list=[],ra:list=[],dec:list=[]):
        """ Generates the data for Method of Gauss calculations
            Args:
                loaded (bool): True if data is already loaded
                selTime (list): list of chosen times (in Julian days) for observations 1,2,3
                selDate (list): list of chosen datesfor observations 1,2,3
                ra (list): list of right ascensions
                dec (list): list of declinations
            Returns:
                lists: ra, dec, R1, R2, R3, taus, ts (original times in Julian days)
        """
        if not loaded:
            self.data.getInput(self.inputFile) # formats and generates info
            # generate data
            if not(selTime==[]): # using Julian day times
                if ra==[] and dec==[]:
                    ra,dec=[],[]
                    for time in selTime:
                        r,d=self.data.getRADECInput(time=time)
                        ra.append(np.deg2rad(r))
                        dec.append(np.deg2rad(d))
                R1=self.data.getSunInput(time=selTime[0])
                R2=self.data.getSunInput(time=selTime[1])
                R3=self.data.getSunInput(time=selTime[2])
                # calculate the taus
                t1,t2,t3=selTime[0],selTime[1],selTime[2]
                ts=[t1,t2,t3]
                T1=t1-t2
                T3=t3-t2
                T=t3-t1
                taus = [T1,T3,T] # in Gaussian days

            elif not(selDate==[]):
                if ra==[] and dec==[]:
                    ra,dec=[],[]
                    for date in selDate:
                        r,d=self.data.getRADECInput(date=date)
                        ra.append(np.deg2rad(r))
                        dec.append(np.deg2rad(d))
                R1=self.data.getSunInput(date=selDate[0])
                R2=self.data.getSunInput(date=selDate[1])
                R3=self.data.getSunInput(date=selDate[2])
                # calculate the taus given the dates
                t1,t2,t3=self.data.getJDTime(selDate[0]),self.data.getJDTime(selDate[1]),self.data.getJDTime(selDate[2])
                ts=[t1,t2,t3]
                T1=t1-t2
                T3=t3-t2
                T=t3-t1
                taus = [T1*self.k,T3*self.k,T*self.k]

            else: raise Exception("No data given")
            self.loadedValues=[ra,dec,np.array(R1),np.array(R2),np.array(R3),taus,np.array(ts)]
        else:
            if not(ra==[] and dec==[]):
                self.loadedValues[0] = ra
                self.loadedValues[1] = dec
        return self.loadedValues
    
    def MoGCalcRhos(self, rhohats:list, coefD:list, fvals:list, gvals:list):
        """ Generates the rhos (and their magnitudes) for Method of Gauss calculations
            Args:
                rhohats (list): the direction vectors for rhos
                coefD (list): the D coefficients
                fvals (list): the f values
                gvals (list): the g values
            Returns:
                lists: rhos, rhomags
        """
        f1,f3=fvals
        g1,g3=gvals
        D0,D11,D12,D13,D21,D22,D23,D31,D32,D33=coefD
        C1=g3/(f1*g3-g1*f3)
        C2=-1
        C3=-g1/(f1*g3-g1*f3)
        rho1=(C1*D11+C2*D12+C3*D13)/(C1*D0)
        rho2=(C1*D21+C2*D22+C3*D23)/(C2*D0)
        rho3=(C1*D31+C2*D32+C3*D33)/(C3*D0)
        rhomags=np.array([rho1,rho2,rho3])
        rhos=np.array([rhohats[0]*rho1, rhohats[1]*rho2, rhohats[2]*rho3])
        return rhos, rhomags
    
    def MoGCalcPos(self, rhos:list, Rs:list)->list:    
        """ Calculates the r vectors (position of asteroid from sun) for Method of Gauss calculations
            Args:
                rhos (list): the position from Earth vectors
                Rs (list): the position from Sun vectors
            Returns:
                lists: rs
        """
        return rhos-Rs
        
    def MoGGetErr(self, prev:list, cur:list, tolerance:float)->bool:
        """ Calculates the r vectors (position of asteroid from sun) for Method of Gauss calculations
            Args:
                prev (list): previous r vector
                cur (list): the newly calculated r vector
                tolerance (float): the tolerance
            Returns:
                bool: True if error is good, False if no within tolerance
        """
        for i in range(len(prev)):
            if abs(prev[i]-cur[i])>tolerance: return False
        return True
    
    def MoGGetAdjustedTaus(self, origt:list, rhomags:float)->list:
        """ Returns adjusted taus for Method of Gauss calculations
            Args:
                origt (list): original observation times in Julian days
                rhomags (list): the magnitude of the rho vectors
            Returns:
                list: the adjusted taus
        """
        ts=np.copy(origt)-rhomags/self.cAU
        t1,t2,t3=ts
        return [(t1-t2)*self.k,(t3-t2)*self.k,(t3-t1)*self.k]
    
    def MoG(self,selTime:list=[],selDate:list=[],override:bool=True,ra:list=[],dec:list=[],loaded:bool=False):
        """ Performs Method of Gauss calculations to determine orbital elements
            Args:
                selTime (list): list of chosen times (in Julian days) for observations 1,2,3
                selDate (list): list of chosen datesfor observations 1,2,3
                override (bool): boolean, True if override info
                ra (list): optional list of right ascensions
                dec (list): optional list of declinations
                loaded (bool): optional, True if all data is already loaded (speeds up run time)
            Returns:
                None
        """
        # generate data
        ra,dec,R1,R2,R3,taus,ts=self.MoGGenData(False,selTime,selDate,ra,dec)
        
        # calculate the initial estimate for r2 DONE
        coefD,rhohats=self.getDCoef(ra, dec, R1, R2, R3)

        SELcoefD=[coefD[0],coefD[4],coefD[5],coefD[6]] # [D0,D11,D12,D13,D21,D22,D23,D31,D32,D33] -> [D0,D21,D22,D23]
        r2MagGuesses,rhoMagGuesses=self.SEL(taus, R2, rhohats[1], SELcoefD)
        
        results=[]
        
        # calculate the f and g values to second order
        for r2mag in r2MagGuesses:
            # initial guesses
            fvals,gvals=self.getFGVals(taus[0], taus[1], r2mag, [], 2, 2) # using r2mag
            rhos,rhomags=self.MoGCalcRhos(rhohats, coefD, fvals, gvals)
            rs=self.MoGCalcPos(rhos, np.array([R1,R2,R3]))
   
            # calculate the central velocity vector
            f1,f3=fvals
            g1,g3=gvals
            r2dot = (-f3/(-f3*g1+g3*f1))*rs[0] + (f1/(f1*g3-f3*g1))*rs[2]
            counter=0
            timedOut=True
            timeout=time.time() + 60*3 # timeout after 3 minutes
            
            while counter<5000 and time.time()<timeout: # timeout
                prev=rs[1]
                
                # time adjustment
                newtaus=self.MoGGetAdjustedTaus(ts, rhomags)
                
                fvals,gvals=self.getFGVals(newtaus[0], newtaus[1], r2mag, r2dot, 4, 4, rs[1]) # using r2mag
                rhos,rhomags=self.MoGCalcRhos(rhohats, coefD, fvals, gvals)
                rs=self.MoGCalcPos(rhos, np.array([R1,R2,R3]))
                
                f1,f3=fvals
                g1,g3=gvals
                r2dot = (-f3/(-f3*g1+g3*f1))*rs[0] + (f1/(f1*g3-f3*g1))*rs[2]
                
                counter+=1
                
                if (self.MoGGetErr(prev, rs[1], 1e-23)): 
                    timedOut=False
                    break
            
            if not timedOut:
                # rotate around x axis to get to ecliptic
                r2=rotX(np.array(rs[1]),-self.eps)
                r2dot=rotX(r2dot,-self.eps)
                rho2=rotX(rhos[1],-self.eps)
                results.append([r2, r2dot, rho2]) # position and velocity
        
        resIndex=0
        if len(results)>1: # get user input to choose which result to use 
            for i in range(len(results)):
                print("Result",str(i)+":")
                print("\tPos from Sun:",r2,"\n\tVel:",r2dot,"\n\tPos from Earth:",rho2,"\n")

            while True:
                try: 
                    resIndex=int(input("Choose which result to use (int): "))
                    if not(0<=resIndex<len(results)): resIndex/=0 # throw an error
                    break
                except:
                    print("Enter a valid integer")
        
        if len(results)==0: return False, 0, 0, 0, 0, 0, 0
            
        # get final answers
        pos,vel,rho=results[resIndex]
        
        if override:
            self.genElements(pos,np.array(vel)*(2*math.pi)/365.2568983,ts[1])
            self.vel = vel
            self.pos = pos
            self.rho = rho
        else:
            a,e,i,o,v,w,T,M=self.genElements(pos,np.array(vel)*(2*math.pi)/365.2568983,ts[1], update=False)
            return True, a,e,i,o,w,T
        
    def getError(self, results:list):
        """ Prints error and results from orbital elements determination after Method of Gauss calculations
            Args:
                results (list): [e,0,i,o,w,T,0,0,0,a]
            Returns:
                None
        """
        # adjust time of perihelion from results to match period of calculated time of perihelion
        period=results[9]**(3/2)*self.toGaussian
        results[5]=self.time-((self.time-results[5])%period) # subtract the change in time since the last perihelion from the obs 2 time 
        self.od.printError(results)
        
    def exportResults(self, time:float, date:str, fileName:str):
        """ Exports results to a file
            Args:
                time (float): time in Julian days for the Mean Anomaly
                date (str): date for Mean Anomaly
                fileName (str): the exported file name
            Returns:
                None
        """
        M=self.od.getTimeMeanAnomaly(time, date)
        self.data.printResults(fileName, self.pos, self.vel/self.toGaussian*(2*math.pi), self.rho, self.a, self.e, self.i, self.o, self.T, self.w, date, M)
        
    def genEphemeris(self, inputFile:str, outputFile:str, actualEph:str):
        """ Generates ephemeris for all times in a given sun position file. Exports to an output file
            Args:
                inputFile (str): input file name
                outputFile (str): output file name
                actualEph (str): file with real values from JPL horizons
            Returns:
                None
        """
        results=[]
        for time,date,R in self.data.getAllSunPos(inputFile):
            ra,dec=self.ephemeris(time, date, sunPos=R)
            ra=str(ra[0]) + " " + str(ra[1]) + " " + str(ra[2])
            if dec[0]<0: sign="-"
            else: sign="+"
            dec=sign+str(dec[0]) + " " + str(dec[1]) + " " + str(dec[2])
            results.append([date,str(time),ra,dec])
        
        self.data.exportEphemeris(outputFile, results, actualEph)
        
    def monteCarlo(self, n:int, files:str, outputFile:str, results:list, selTime:list=[], selDate:list=[]):
        """ Using uncertainty, runs through Method of Gauss n-times to generate orbital elements. 
            Outputs to file.
            Args:
                n (int): number of times to run monteCarlo
                files (list): list of fits file names
                outputFile (str): the output file name
                results (list): the actual values for error analysis [a, e, i, o, w, T]
                selTime (list): optional; the selected times for which to run Method of Gauss
                selDate (list): optional; the selected dates for which to run Method of Gauss
            Returns:
                None
        """
        rmsDec,rmsRA=[],[]
        for file in files:
            data = fits.open(file)[1].data
            
            fieldDec, indexDec = data.field_dec[:], data.index_dec[:]
            fieldRA, indexRA = data.field_ra[:], data.index_ra[:]
            
            dec = np.sqrt(1/int(fieldDec.shape[0]) * sum([(indexDec[i]-fieldDec[i])**2 for i in range(int(fieldDec.shape[0]))]))
            ra = np.sqrt(1/int(fieldRA.shape[0]) * sum([(indexRA[i]-fieldRA[i])**2 for i in range(int(fieldDec.shape[0]))]))
            
            rmsDec.append(dec)
            rmsRA.append(ra)
            
            
        ora, odec, R1, R2, R3, taus, ts = self.MoGGenData(False,selTime=selTime,selDate=selDate)
        
        vals=[]
        num=0
        for trial in range(n):
            newdec = [np.deg2rad(np.random.normal()*rmsDec[i] + np.rad2deg(odec[i])) for i in range(3)]
            newra = [np.deg2rad(np.random.normal()*rmsRA[i] + np.rad2deg(ora[i])) for i in range(3)]
            works, a, e, i, o, w, T=self.MoG(selTime=selTime,selDate=selDate,override=False,ra=newra,dec=newdec,loaded=True)
            if works:
                num+=1
                calcVals=[num,a,e,i,o,w,T] # sus
                vals.append(calcVals)
                vals.append(calcVals)
                
        # adjust time of perihelion for results
        # for time of perihelion, need to adjust to be close to the given observation time
        # because our calculated time of perihelion is closest to the given observation time
        period=results[0]**(3/2)*self.toGaussian
        results[5]=ts[1]-((ts[1]-results[5])%period) # subtract the change in time since the last perihelion from the obs 2 time 
                
            
        # write to output file
        self.data.exportMonteCarlo(outputFile, vals.copy(), results)
        
        # histograms
        labels=["","Semi-Major Axis","Eccentricity", "Inclination", "Longitude of Ascending Node", "Argument of Perihelion", "Time of Perihelion Passage"]
        xlabels=["AU","Value","Degrees","Degrees","Degrees","Julian Day Number"]
        custom_lines = [Line2D([0], [0], color='black', linestyle='dashed',lw=1), Line2D([0], [0], color='blue', lw=1)]
  
        for j in range(5):
            mean=sum([vals[i][1+j] for i in range(len(vals))])/len(vals)
            sdom=np.sqrt(sum([(vals[i][j+1]-mean)**2 for i in range(len(vals))])/len(vals))
            
            plt.hist([vals[i][1+j] for i in range(len(vals))],bins=20)
            plt.axvline(results[j], color='k', linestyle='dashed', linewidth=1)
            plt.axvline(mean, color='b', linewidth=1)
            plt.title(labels[1+j])
            plt.legend(custom_lines, ['True Value', 'Mean'])
            plt.xlabel(xlabels[j])
            plt.ylabel("Frequency")
            plt.show()

            print(labels[j+1]+": \n\tmean: "+str(mean)+"\n\terror: " + str(error(results[j],mean)) + "\n\tstandard deviation of mean: " + str(sdom) + "\n")

        # special case for time of perihelion
        mean=sum([vals[i][6] for i in range(len(vals))])/len(vals)
        sdom=np.sqrt(sum([(vals[i][6]-mean)**2 for i in range(len(vals))])/len(vals))
            
        plt.hist([vals[i][6] for i in range(len(vals))],bins=20)
        plt.axvline(results[5], color='k', linestyle='dashed', linewidth=1)
        plt.axvline(mean, color='b', linewidth=1)
        plt.title(labels[6])
        plt.legend(custom_lines, ['True Value', 'Mean'])
        plt.xlabel(xlabels[5])
        plt.ylabel("Frequency")
        plt.show()

        print(labels[6]+": \n\tmean: "+str(mean)+ "\n\terror: " + str(error(results[5],mean)) + "\n\tstandard deviation of mean: " + str(sdom) + "\n")

 
# final functions
def RunCompleteOD(iterations:int, inputFile:str, fitsFiles:list, sunFile:str, dates:list, results:list, actualEph:str):
    """ Runs complete orbit determination code. Generates three files:
            SoongODResults.txt (the results of orbital determination),
            SoongMonteCarloOutput.txt (the results from the Monte Carlo simulation),
            SoongGeneratedEphemeris.txt (the results from the ephemeris generation)
        Args:
            iterations (int): the number of iteration for which to run the Monte Carlo simulation
            inputFile (str): path for file containing observations to determine orbital elements
            fitsFiles (list): list of strings describing paths for the three fits files for Monte Carlo sim
            sunFile (str): path for file containing times and sun positions for ephemeris generation
            dates (list): list of dates for orbital elements determination
            results (list): the actual values for orbital elements in the format [a,e,i,o,w,m,T] 
            actualEph (str): the file path for the actual ephemeris from JPL Horizons
        Returns:
            None
    """
 
    # generate orbital elements
    data=Data()
    od=OD(inputFile)
    od.MoG(selDate=dates)
    
    # determine error
    a,e,i,o,w,m,T = results
    print("--- Results from orbital elements determination ---")
    od.getError([e,0,i,o,w,T,0,0,0,a])
    od.exportResults(2459784.7916667, "July 24, 2022 7:00 UT", "SoongODResults.txt") 
    
    # Monte Carlo
    results = [a,e,i,o,w,T] 
    print("\n--- Monte Carlo Simulation ---")
    od.monteCarlo(iterations, fitsFiles, "SoongMonteCarloOutput.txt", results, selDate=dates)
    
    # generate the ephemeris
    od.genEphemeris(sunFile, "SoongGeneratedEphemeris.txt", actualEph)
    print("--- Ephemeris generation completed ---")

    
def GenerateEphemeris(inputFile:str, ODdates:list, sunFile:str, actualEph:str):
    """ Outputs ephemeris to file. Generates one file:
            GeneratedEphemeris.txt (the results of ephemeris generation)
        Args:
            inputFile (str): path for file containing observations to determine orbital elements
            ODdates (list): list of dates for orbital elements determination
            sunFile (str): path for file containing times and sun positions for ephemeris generation
            actualEph (str): path for file containing real ephemeris results from JPL Horizons
        Returns:
            None
    """
 
    # calculating orbital elements
    data=Data()
    od=OD(inputFile)
    od.MoG(selDate=ODdates)
    
    # generate ephemeris
    od.genEphemeris(sunFile, "GeneratedEphemeris.txt", actualEph)
    
    print("Finished generating ephemeris")