import sys
import os
from pathlib import Path
import numpy as np
import math
import h5py
import pyvista as pv
import matplotlib.pyplot as plt
import DMD
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

print('About to print graphs')
results=sys.argv[1] #eg. case_043_low/results/
file_stride = sys.argv[2] #number of files to skip

dd = DMD.Dataset(Path((results + os.listdir(results)[0])), file_stride=file_stride)
dd = dd.assemble_mesh()

#make SVD graphs
data = h5py.File('DMD_files/DMD.h5','r')
print('obtained data')
Sigma = np.absolute(np.diagonal(np.array(data['Sigma'])))
Lambda = np.array(data['Lambda'])
L_R = np.real(Lambda)
L_I = np.imag(Lambda)

phi = np.array(data['phi'])
D = np.array(data['D'])
modes = phi@D 
mode_200 = np.linalg.norm(np.real(modes[:,200]).reshape((-1,3)),axis=1) #choose the 50th mode and take its magnitude
mode_100 = np.linalg.norm(np.real(modes[:,100]).reshape((-1,3)),axis=1) #choose the 50th mode and take its magnitude
mode_0 = np.linalg.norm(np.real(modes[:,0]).reshape((-1,3)),axis=1)

dd.mesh.point_data['mode_200']=mode_200
dd.mesh.point_data['mode_100']=mode_100
dd.mesh.point_data['mode_0']=mode_0
dd.mesh.save('DMD_files/mode_shapes.vtu')

N = len(Sigma)
print('plotting...')
fig1, ax1 = plt.subplots(1,1, figsize=(5,4))
ax1.scatter(range(N), Sigma)
ax1.set_xlim([0, N])
ax1.set_yscale("log")
title1 = '$\Sigma$'
ax1.set_title(title1)
plt.tight_layout
plt.savefig('DMD_files/Sigma.png')

circle1=plt.Circle((0,0),1, fill=False, ls='--')
fig2, lax1 = plt.subplots(1,1, figsize=(4,4))
lax1.scatter(L_I, L_R)
lax1.add_patch(circle1)
title1 = '$Ritz$'
lax1.set_title(title1)
lax1.set_xlabel('$\mathbb{Re}(\mu)$')
lax1.set_ylabel('$\mathbb{Im}(mu)$')
lax1.set_xlim([-1.5, 1.5])
lax1.set_ylim([-1.5, 1.5])
plt.tight_layout
plt.savefig('DMD_files/Ritz.png')


