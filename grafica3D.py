import numpy as np
import matplotlib.pyplot as plt


# queremos modelar la función r(t) = Acos(wt) i + Asin(wt) j + Bt k

A = 1
B = 3
omega = 2*np.pi/10
N = 50

# puntos temporales
t = np.linspace(0,10,N)
# puntos de posición
r = np.zeros((N,3))

r[:,0] = A*np.cos(omega*t)
r[:,1] = A*np.sin(omega*t)
r[:,2] = B*t

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111,projection='3d')

ax.scatter(r[:,0],r[:,1],r[:,2], color = 'k', marker='o', s=100)
ax.view_init()