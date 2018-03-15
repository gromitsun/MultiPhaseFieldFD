import numpy as np
from scipy.ndimage.filters import laplace


nx = 256
ny = 256
nz = 256

# Equilibrium compositions at T = 883 K
xeq1 = 0.00839458938258
xeq2 = 0.079106855694

# Equilibrium compositions at T = 836.7 K
xeq1 = 0.020151064684394093
xeq2 = 0.15325846495183987

# Equilibrium compositions from Jin (at T = 826.15 K?)
xeq1 = 0.019862472879877200
xeq2 = 0.1544897158058190

# Equilibrium compositions at T = 837.7 K (parabolic from matlab)
xeq1 = 0.0196414793000061
xeq2 = 0.15207343594726

x = np.arange(nx) - nx/2
y = np.arange(ny) - ny/2

xx, yy = np.meshgrid(x, y)


# disk
bw = np.sqrt(xx**2+yy**2)<=50

# planar interface
bw = xx < 0

phi = np.zeros(shape=(ny,nx))
comp = np.zeros(shape=(ny,nx))

# sharp
phi[bw] = 1

comp[bw] = xeq1
comp[np.logical_not(bw)] = xeq2



# # smooth
# tanhxx = (np.tanh(200.0*xx/x.max())+1)/2
# phi = tanhxx
# comp = xeq1+(xeq2-xeq1)*tanhxx

# # smooth by diffusion
# dt = 1e-2
# n = 0
# for i in range(n):
#     phi += dt*laplace(phi)
#     comp += dt*laplace(comp)


phi = np.repeat(phi.reshape(1,ny,nx), nz, axis=0)
comp = np.repeat(comp.reshape(1,ny,nx), nz, axis=0)


phi.tofile('phia.bin')
(1-phi).tofile('phib.bin')
comp.tofile('comp.bin')


import matplotlib.pyplot as plt

plt.figure()
plt.imshow(phi[nz/2])
plt.colorbar()
plt.figure()
plt.imshow(comp[0])
plt.colorbar()
# plt.figure()
# plt.plot(x,tanhxx[0])
plt.figure()
plt.plot(phi[nz/2,ny/2,:])
plt.plot(comp[nz/2,ny/2,:])
plt.show()
