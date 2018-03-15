import numpy as np
import matplotlib.pyplot as plt

step = 1000000

datadir= '/Users/yue/Library/Developer/Xcode/DerivedData/MoelansPhaseFieldFD-dkgbyukpiwflguerltsdaasmoecd/Build/Products/Debug/output/'
datadir= 'output/'
f1 = datadir + 'comp_' + str(step) +'.bin'
f2 = datadir + 'phi_' + str(step) +'.bin'

a1 = np.fromfile(f1).reshape(256,256)
a2 = np.fromfile(f2).reshape(256,256)


plt.figure()
h1, = plt.plot(a1[128,:])

plt.figure()
h2, = plt.plot(a2[128,:])

# plt.legend([h1, h2], ['comp', 'phi'], loc='best')


# plt.figure()
# plt.imshow(a1)
# plt.colorbar()

# plt.figure()
# plt.imshow(a2)
# plt.colorbar()


plt.show()