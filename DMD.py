#Compute the Dynamic Mode Decomposition of an Unsteady Flow - Implementation by ALK Haley
# Need to calculate W, Sigma^-1, Y for the modes (Phi)
# Need to calculate Y_star, Sigma, W_star,T^-1 for the amplitudes (D)
# Get W and Sigma from SVD
# Get S_bar from U_star*Psi'*W*Sigma^-1
# Get eigenvalues (Lambda) and eigenvectors (Y) of S_bar
# multiply U*Y to get the modes
# Get the amplitudes with D = Y_star*Sigma*W_star*T^-1
import sys
import os
from pathlib import Path
import numpy as np
#import scipy.linalg as la
import math
import h5py
import pyvista as pv
import vtk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

if __name__=="__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

class Dataset():
    """ Load BSL-specific data and common ops. 
    """
    def __init__(self, folder, mesh_folder = None, file_glob_key=None, file_stride=1, mono=False,random=False):
        self.folder = Path(folder)
        self.mono = mono
        self.random=random

        if self.mono and random==False: #Oasis output
            if file_glob_key is None:
                file_glob_key = '*.h5'
            self.result_files = sorted(folder.glob(file_glob_key), key=self._get_results)
            #print(self.result_files)
            self.tsteps = 0
            up_files=[]
            for file in self.result_files:
                with h5py.File(file,'r') as hf:
                    new = [file] * len(list(hf['VisualisationVector']))
                    up_files.extend(new)
            self.up_files=up_files
            self.tsteps = 1+math.floor((len(self.up_files)-1)/file_stride)
            self.mesh_file = self.result_files[0]
            self.file_stride=file_stride
        elif self.random:
            self.up_files = list(folder.glob('*all*'))
            self.tsteps=1000
            self.mesh_file = list((folder / ('../data')).glob('*mesh*'))[0] 
            self.file_stride=file_stride
        else:
            if file_glob_key is None:
                file_glob_key = '*_curcyc_*up.h5'
            mesh_glob_key = '*h5'
            if mesh_folder is None:
                self.mesh_file = sorted(folder.glob(mesh_glob_key), key=lambda x: len(x.stem))[0]
                self.ts = '_ts='
                self.tssplit = '_'
                self.different_folders=False
            else:
                self.mesh_file = sorted(mesh_folder.glob(mesh_glob_key), key=lambda x: len(x.stem))[0]   
                self.ts='_tstep=' 
                self.tssplit='.'
                self.different_folders=True

            self.up_files = sorted(folder.glob(file_glob_key), key=self._get_ts)[::int(file_stride)]
            self.tsteps = len(sorted(folder.glob(file_glob_key), key=self._get_ts)[::int(file_stride)])
            self.times = sorted(folder.glob(file_glob_key), key=self._get_time)[::int(file_stride)]

    def __call__(self, idx, array='u', file=None):
        if self.mono:
            ndx = idx*self.file_stride #the actual tstep we would want with no file_stride
            file = self.up_files[ndx] #the file that contains this tstep            
            with h5py.File(file,'r') as hf:
                #print(file, ndx,self._get_results(file))
                ts = int(ndx-self._get_results(file)/10) #the actual id of the tstep in this file
                val = np.array(hf['VisualisationVector'][str(ts)])
        elif self.random:
            file=self.up_files[0]
            ndx = idx*self.file_stride
            with h5py.File(file,'r') as hf:
                val = np.array(hf['u'][str(ndx)])
        else:
            """ Return velocity in u_file. """
            if array in ['u', 'p']:
                h5_file = self.up_files[idx]
                with h5py.File(h5_file, 'r') as hf:
                    if self.different_folders:
                        val = np.array(hf[array])
                    else:
                        val = np.array(hf['Solution'][array])
            else:
                h5_file = file
                with h5py.File(h5_file, 'r') as hf:
                    val = np.array(hf[array])
        return val
    
    def _get_results(self, h5_file):
        return int(h5_file.stem.split('_')[-1])

    def _get_ts(self, h5_file, ts=None):
        """ Given a simulation h5_file, get ts. """
        if self.mono or self.random:
            return ts
        else:
            return int(h5_file.stem.split(self.ts)[1].split(self.tssplit)[0])
    
    def _get_ts_swirl(self, h5_file):
        """ Given a different h5_file, get ts. """
        return int(h5_file.stem.split('_')[1].split('.')[0])
        
    def _get_time(self, h5_file):
        """ Given a simulation h5_file, get time. """
        return float(h5_file.stem.split('_t=')[1].split('_')[0]) / 1000.0
    
    def check_cells(self):
        cellsize=self.mesh.compute_cell_sizes().cell_data['Volume']
        neg=np.flatnonzero(cellsize<0)
        n3=5*neg+2
        n4=5*neg+3
        cells = self.mesh.cells.copy()
        cells_n3=cells[n3].copy()
        cells_n4=cells[n4].copy()
        cells[n3]=cells_n4
        cells[n4]=cells_n3
        self.mesh.cells = cells
        cellsize=self.mesh.compute_cell_sizes().cell_data['Volume']
        new_neg=np.flatnonzero(cellsize<0)
        if len(new_neg)>0:
            print("check_cells didn't work!")
            sys.exit()

    def assemble_mesh(self):
        """ Create UnstructuredGrid from h5 mesh file. """
        assert self.mesh_file.exists(), 'mesh_file does not exist.'
        with h5py.File(self.mesh_file, 'r') as hf:
            if self.random==False:
                if self.mono:
                    points = np.array(hf['Mesh']['0']['mesh']['geometry'])*(10**-3)
                    cells = np.array(hf['Mesh']['0']['mesh']['topology'])
                else:
                    points = np.array(hf['Mesh']['coordinates'])*(10**-3)
                    cells = np.array(hf['Mesh']['topology'])

                celltypes = np.empty(cells.shape[0], dtype=np.uint8)
                celltypes[:] = vtk.VTK_TETRA
                cell_type = np.ones((cells.shape[0], 1), dtype=int) * 4
                cells = np.concatenate([cell_type, cells], axis = 1)
                self.mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
                self.surf = self.mesh.extract_surface()
                self.check_cells()
            else:
                #points = np.array(hf['Mesh']['coordinates'])
                #cells = np.array(hf['Mesh']['topology'])
                #celltypes = np.empty(cells.shape[0], dtype=np.uint8)
                #celltypes[:] = vtk.VTK_HEXAHEDRON
                #cell_type = np.ones((cells.shape[0], 1), dtype=int) * 8
                #cells = np.concatenate([cell_type, cells], axis = 1)
                #self.mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
                xrng=np.linspace(math.pi,2*math.pi,64)
                x,y,z=np.meshgrid(xrng, xrng, xrng, indexing='ij')
                self.mesh = pv.StructuredGrid(x,y,z)
                self.surf = self.mesh.extract_surface()
                #print("Before exit")
                #exit(1)
            # self.assemble_surface()
        return self

#Note: Only need to calculate upper part of matrix due to symmetry
def compute_IP_mat(dd, pieces, last_piece, rows, IP_mat_n):
    """
    input:
        dd is the dataset object
        #NO idxs is a list of matrix indices to calculate
        IP_mat is the inner product matrix mxm
    """
    #for i in range(len(idxs)):
    #    u_m = dd(idxs[i,0]).flatten() 
    #    u_mi = dd(idxs[i,1]).flatten()
    #    IP_mat_n[i] = np.sum(u_m*u_mi)
    if rank<size-1:
        last = (rank+1)*pieces
    else:   
        last = rank*pieces+last_piece
    ii = 0
    for i in range(rank*pieces,last):
        xi = dd(i).flatten()
        for j in range(rows):
            xj = dd(j).flatten()
            IP_mat_n[ii,j]=np.sum(xi*xj)
        ii +=1
    #print('IP_mat_{}'.format(rank),IP_mat_n)

def WIP_eigs(IP_mat):
    Sigma_2, W = np.linalg.eig(IP_mat)
    Sigma = np.sqrt(Sigma_2)
    return np.diag(Sigma), W # both mxm

def compute_sv(W,Sigma):
    return W@np.linalg.inv(Sigma) #mxm

def compute_U(U, sv, u_m, m):
    U += u_m.reshape((-1,1))*sv[m,:].reshape((1,-1)) #the summation over all snapshots (nxm)

def reconstruct_DMD(phi,d,alpha,omega, tsteps, outfolder):
    T = 0.915
    step = T/tsteps
    t = 0
    u = np.zeros((len(phi),tsteps))
    for ts in range(tsteps):
        t=ts*step
        u = phi@d@np.exp(alpha*t+1j*omega*t)
        u_3d=u.reshape((-1,3))
        with h5py.File(outfolder + '/u_t={}_tstep={}.h5'.format(t, ts),'w') as f:
            f.create_dataset(name='u',data=u_3d)

def gen_modes_strengths(U_0, S_bar_0, Sigma_0, W_0, dt, rows, tsteps):
    Lambda_0, Y_0 = np.linalg.eig(S_bar_0)
    #print('Y_{} = '.format(direc), Y_0)
    T_0 = np.vander(Lambda_0, N=rows, increasing=True) #vandermonde matrix
    #print('T_{}='.format(direc),T_0)
    D_0 = np.matmul(Y_0.conj().T,np.matmul(Sigma_0,np.matmul(W_0.conj().T,np.linalg.inv(T_0)))) #amplitudes contained in columns (conj().T is conjugate transpose) (mxm)
    phi_0 = U_0@Y_0 #rows are the modes(nxm)
    omega = np.array(range(rows))/dt
    alpha_0 = np.log(np.absolute(Lambda_0))
    print('Printing DMD to file...')
    with h5py.File('DMD_files/DMD.h5','w') as f:
        f.create_dataset(name='alpha',data=alpha_0)
        f.create_dataset(name='omega', data=omega)
        f.create_dataset(name='phi', data=phi_0)
        f.create_dataset(name='D', data=D_0)
        f.create_dataset(name='Lambda', data=Lambda_0)
        f.create_dataset(name='T', data=T_0)
        f.create_dataset(name='Sigma', data=Sigma_0)
        f.create_dataset(name='W', data=W_0)
        f.create_dataset(name='U', data=U_0)
    #now reconstruct the first four modes:
    print('Reconstructing DMD to files')
    reconstruct_DMD(phi_0[:,:4],D_0[:4,:4],alpha_0[:4], omega[:4], tsteps, outfolder)


if __name__=="__main__":
    results=sys.argv[1] #eg. case_043_low/results/
    case_name=sys.argv[2] #eg. case_043_low
    file_stride = sys.argv[3] #number of files to skip

    dd = Dataset(Path((results + os.listdir(results)[0])), file_stride=file_stride)
    splits = case_name.split('_')
    seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]
    main_folder = Path(results).parents[0]
    dd = dd.assemble_mesh()
    
    rows = len(dd.up_files)-1 #because Psi and Psi' are one tstep shifted from each other
    cols = len(dd(0))*3
    dt = dd._get_ts(dd.up_files[1])-dd._get_ts(dd.up_files[0])

    #total_entries = int(rows*rows/2+rows/2) #total inner product matrix entries (symmetric)
    #indices = np.zeros((total_entries, 2), dtype=np.uint8)
    #i = 0
    #for m in range(rows):
    #    for mi in range(m, rows):
    #        indices[i]= np.array([m, mi]) 
    #        i +=1
    total_entries=rows*rows
    pieces = math.floor(rows/(size-1))
    last_piece = rows-pieces*(size-1) 
    #print(pieces)
    #print(last_piece)
    #Compute inner product matrix (IP_mat) on 40 procs
    #Note: this is currently inefficient because IP_mat is symmetric, so technically we can just calculate the triangular matrix and then fix it later.
    if rank == 0:
        data = np.zeros((rows,rows), dtype=np.double)
        for s in range(1,size):
            if s<(size-1):
                data_0=np.empty(pieces*rows, dtype=np.double)
                comm.Recv(data_0, source=s, tag =101)
                data[s*pieces:(s+1)*pieces, :]=data_0.reshape((pieces,rows))
            else:
                data_0=np.empty(last_piece*rows, dtype=np.double)
                comm.Recv(data_0, source=s, tag =101)
                data[s*pieces:s*pieces+last_piece, :]=data_0.reshape((last_piece,rows))
            #print('IP_mat_recvd_{}='.format(s), data_0)

        #compute on this proc too
        #indices_n = indices[rank*pieces:(rank+1)*pieces]
        IP_mat_0 = np.zeros((pieces,rows))
        compute_IP_mat(dd,pieces, last_piece, rows, IP_mat_0)
        #put into data array
        data[rank*pieces:(rank+1)*pieces, :]=IP_mat_0
        #print(data)
        #Arrange inner product matrix
        IP_mat=np.zeros((rows,rows))
        IP_mat = data #put data into matrix
        #IP_mat = IP_mat + IP_mat.T - np.diag(np.diagonal(IP_mat)) #make symmetric matrix
        #print('IPmat=',IP_mat)
        #compare
        #X= np.zeros((cols,rows))
        #for i in range(rows):
        #    X[:,i]=dd(i).flatten()
        #print('XX=',X.T@X)

        #Calculate some things in serial
        Sigma, W = WIP_eigs(IP_mat)
        sv = compute_sv(W,Sigma)
        print('Computed singular value decomposition!', flush=True) #Note that sv is just a convenient matrix to store for later computation
        #print(sv_x)
    else:
        if rank<(size-1):
            #indices_n = indices[rank*pieces:(rank+1)*pieces]
            IP_mat_n = np.zeros((pieces,rows))
        else:
            #indices_n = indices[rank*pieces:rank*pieces+last_piece]
            IP_mat_n = np.zeros((last_piece,rows))
        compute_IP_mat(dd, pieces, last_piece, rows, IP_mat_n)
        comm.Send(IP_mat_n.flatten(), dest=0, tag = 101)
        #print('IP_mat_{}='.format(rank), IP_mat_n)

        print('Computed inner product matrix on processor {}!'.format(rank), flush=True)
        sv = np.empty([rows,rows], dtype=np.complex128) #sv should be a real matrix because sigma is real?

    comm.Bcast(sv, root=0)
    #print('sv_x[0,3]={} on processor {}'.format(sv_x[0,3],rank))
    #Compute the U matrix on 40 (or however many) procs using the sv matrix
    U = np.zeros((cols,rows), dtype=np.complex128)
    chunk = math.floor(rows/(size-1))
    last_chunk = rows-chunk*(size-1)
    #print(last_chunk)
    #divide up sv matrices by cols to each proc:
    if rank<(size-1):
        lst = range(rank*chunk,(rank+1)*chunk)
        sv_0 = sv[:,lst]
        U_0= np.zeros((cols,chunk), dtype=np.complex128)
    else:
        lst = range(rank*chunk,rank*chunk+last_chunk) 
        sv_0 = sv[:,lst]
        U_0= np.zeros((cols,last_chunk), dtype=np.complex128)
    for m in range(rows):
        u_m = dd(m).flatten()
        u_m1 = dd(m+1).flatten()
        #if (rank ==0) and (m%10==0):
        #    print('U matrix {}% complete'.format(int(100*(m/rows))))
        compute_U(U_0, sv_0, u_m,m)
    print('Computed U matrix on processor {}!'.format(rank), flush=True)

    if rank == 0:
        U[:,lst]=U_0
        for s in range(1,size):            
            if s<(size-1):
                data_r=np.empty((cols,chunk), dtype=np.complex128)
                comm.Recv(data_r, source=s, tag=1)
                U[:,s*chunk:s*chunk+chunk]=data_r
            else:
                data_r=np.empty((cols,last_chunk), dtype=np.complex128)
                comm.Recv(data_r, source=s, tag = 1)
                U[:,s*chunk:s*chunk+last_chunk]=data_r
        print('Received U chunks.', flush=True)
        
    else:
        comm.Send(U_0, dest=0, tag=1)
    #Now we just need to get the time-shifted IP_mat PP' by using the old IP_mat but adding on a column
    if rank==0:
        #compute last column of psi*psi' matrix
        xj=np.zeros((rows,1))
        um=dd(rows).flatten() #the last snapshot
        for idx in range(len(dd.up_files)-1):
            umv=dd(idx).flatten()
            xj[idx] = np.sum(umv*um)
        IP_mat_p=np.concatenate((IP_mat[:,1:],xj), axis=1)

        #construct S_bar
        S_bar = np.matmul(np.linalg.inv(Sigma),np.matmul(W.conj().T,np.matmul(IP_mat_p,np.matmul(W,np.linalg.inv(Sigma)))))
        
        outfolder = 'DMD_files/u_files'
        if not Path(outfolder).exists():
            Path(outfolder).mkdir(parents=True, exist_ok=True)

        print('Generating modes and strengths...', flush=True)
        gen_modes_strengths(U, S_bar, Sigma, W, dt, rows, dd.tsteps)

