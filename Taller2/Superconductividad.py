#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy import integrate


# In[7]:


n = 50
x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)   
def GetLegendre(n,x,y):
    
    y = (x**2 - 1)**n
    
    poly = sym.diff( y,x,n )/(2**n * np.math.factorial(n))
    
    return poly
    
Legendre = []
DLegendre = []

for i in range(n+1):
    
    Poly = GetLegendre(i,x,y)
    Legendre.append(Poly)
    DLegendre.append( sym.diff(Poly,x,1) )


# In[8]:




def GetNewton(f,df,xn,itmax=10000,precision=1e-14):
    
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(xn)
            
            error = np.abs(f(xn)/df(xn))
            
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        return False
    else:
        return xn
def GetRoots(f,df,x,tolerancia = 14):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)
        
        if root != False:
            
            croot = np.round( root, tolerancia )
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots


# In[9]:



def GetAllRoots(n,xn,Legendre,DLegendre):

poly = sym.lambdify([x],Legendre[n],'numpy')
Dpoly = sym.lambdify([x],DLegendre[n],'numpy')
Roots = GetRoots(poly,Dpoly,xn)

return Roots


xn = np.linspace(-1,1,100)
Roots = GetAllRoots(n,xn,Legendre,DLegendre)

def GetWeights(Roots,DLegendre):

Dpoly = sym.lambdify([x],DLegendre[n],'numpy')
Weights= 2/( (1-Roots**2)*Dpoly(Roots)**2 )

return Weights
Weights = GetWeights(Roots,DLegendre)
print(Weights)


# In[23]:


def funcion(x,w,t,delta_t):
    H=np.tanh((x**2+delta_t**2))
    J=(x**2+delta_t**2)**1/2
    r=0.5*(H**1/2)*(300/(2*t))/(J*w)
    return sum(r)


# In[25]:


N0V=0.3
delta_t=0.0004
for t in np.arange(1,20,delta_t):
    I=funcion(Roots,Weights,t,delta_t)
    if np.abs(I-1/(N0V)) < delta_t:
        print(t)


# In[ ]:




