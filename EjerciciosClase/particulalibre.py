import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
# from tqdm import tqdm


class Particle:
    
    def __init__(self,r0,v0,a0,t,m=1,radius=1.0,Id=0):
        
        self.dt = t[1] - t[0]
        
        self.r = r0
        self.v = v0
        self.a = a0
        
        self.R = np.zeros((len(t),len(r0)))
        self.V = np.zeros_like(self.R)
        self.A = np.zeros_like(self.R)
        
        self.radius = radius
        self.m = m

    def Evolution(self,i):
        
        self.SetPosition(i)
        self.SetVelocity(i)
        
        self.r += self.v*self.dt 
        
        
    def SetPosition(self, i):
        self.R[i] = self.r
        
    def GetPosition(self,scale=1):
        return self.R[::scale]
    
    def SetVelocity(self, i):
        self.V[i] = self.v
        
    def GetVelocity(self,scale=1):
        return self.V[::scale]

def RunSimulation(t):
    
    r0 = np.array([0.1,0.1])
    v0 = np.array([1,5])
    a0 = np.array([0,0])
    
    p1 = Particle(r0,v0,a0,t)
    
    for it in (range(len(t))):
        p1.Evolution(it)
        
    return p1

dt = 0.01
tmax = 1
t = np.arange(0,tmax,dt)
Particles = RunSimulation(t)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

def init():
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    
def Update(i):
    ax.clear()
    init()
    
    x = Particles.GetPosition()[i,0]
    y = Particles.GetPosition()[i,1]
    
    circle = plt.Circle((x,y),Particles.radius, fill=True)
    ax.add_patch(circle)
    anim.save()

Animation = anim.FuncAnimation(fig,Update,frames=len(t),init_func=init)   