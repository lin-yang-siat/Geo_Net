from scipy.ndimage import gaussian_filter
# import GeodisTK
import numpy as np
import time
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib
import skfmm
import scipy.ndimage
# from scipy.ndimage.filters import median_filter

import nibabel as nib
import statistics
from scipy.io import savemat

def calculate_geodesic_dist3D(z, X, Y,Z):
    z = np.array(z)
    z1 = (z - z.mean()) / z.std()
    # print(z1.max())

    beta = 1
    R = np.zeros(np.shape(z))

    R[X, Y, Z] = 1

    D_E = scipy.ndimage.distance_transform_edt(1 - R)  # 欧几里得距离

    # D_E = D_E / np.max(D_E.flatten())  # 欧几里得距离
    z1 = gaussian_filter(z1, sigma=1)
    gx, gy, gz = np.gradient(z1)
    nab_z = np.sqrt(gx ** 2 + gy ** 2+ gz ** 2)

    # f = (1.0e-3) * np.ones(np.shape(D_E)) + beta*(1-z)*D_E * nab_z ** 2
    # f = beta * (1 - z) * D_E * nab_z ** 2
    f = beta * (np.max(z1) - z1) * nab_z ** 2
    # print('f',np.max(f))

    # f = (f-np.min(f.flatten()))/(np.max(f.flatten())-np.min(f.flatten()))
    f = (1./(f+0.001))#+0.01

    T = skfmm.travel_time(R-0.5*np.ones(np.shape(R)), speed=f, dx=1.0/np.shape(R)[0], order=1)
    T = T/np.max(T)
    return T

def find_centroid(PET_nii_GT):
    X, Y, Z = pet_data.shape
    Position_PET = []
    for x in range(X):
        if np.any(PET_nii_GT[x, :, :] == 1):
            Position_PET.append(x)
    Len_X = int(statistics.median(Position_PET))
    Position_PET = []
    for y in range(Y):
        if np.any(PET_nii_GT[:, y, :] == 1):
            Position_PET.append(y)
    Len_Y = int(statistics.median(Position_PET))
    Position_PET = []
    for z in range(Z):
        if np.any(PET_nii_GT[:, :, z] == 1):
            Position_PET.append(z)
    Len_Z = int(statistics.median(Position_PET))

    return Len_X,Len_Y,Len_Z
def dfs(grid, x, y, z, value, visited, region):
    directions = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    grid_shape = grid.shape
    stack = [(x, y, z)]
    while stack:
        cur_x, cur_y, cur_z = stack.pop()
        for dx, dy, dz in directions:
            nx, ny, nz = cur_x + dx, cur_y + dy, cur_z + dz
            if (0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1] and 0 <= nz < grid_shape[2] and
                grid[nx, ny, nz] == value and (nx, ny, nz) not in visited):
                visited.add((nx, ny, nz))
                region.append((nx, ny, nz))
                stack.append((nx, ny, nz))

def find_connected_regions(grid, value=1):
    connected_regions = []
    visited = set()
    X,Y,Z=grid.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if grid[x, y, z] == value and (x, y, z) not in visited:
                    region = []
                    dfs(grid, x, y, z, value, visited, region)
                    connected_regions.append(region)
    return connected_regions

def calculate_centroid(region):
    if not region:
        return None
    region_array = np.array(region)
    centroid = np.mean(region_array, axis=0)
    return centroid
if __name__ == '__main__':
    import os

    def norm(x):
        x=(x-x.min())/(x.max()-x.min())
        return x
    def theresold(x):
        x[x>0.5]=1
        x[x <= 0.5] = 0
        return x

    base_path = r'D:\Data\HECKTOR_224'
    file_name = os.listdir(os.path.join(base_path))
    Len = len(file_name)
    Num_center=np.zeros((Len,1))
    Acc_center=np.zeros((Len,1))

    ##############3D

    for f in range(Len):
        path=os.path.join(base_path,file_name[f])
        print(path)

        # PET_file=os.path.join(path,'SUV.nii')
        # PET_file_GT =os.path.join(path,'SUV_S.nii')
        PET_file = os.path.join(path, file_name[f]+'_pt.nii.gz')
        PET_file_GT = os.path.join(path, file_name[f]+'_ct_gtvt.nii.gz')
        PET_nii = nib.load(PET_file)
        affine_matrix = PET_nii.affine
        PET_nii_GT = nib.load(PET_file_GT).dataobj
        PET_nii_GT = PET_nii_GT[2:142,2:142,2:142]


        pet_data = PET_nii.get_fdata()
        pet_data = pet_data/pet_data.max()
        pet_data = pet_data[2:142,2:142,2:142]
        pet_data_mean = np.mean(pet_data)
        # pet_data_iter = pet_data
        #初始点
        p_x, p_y, p_z = 10,10,10

        D=calculate_geodesic_dist3D(pet_data, p_x,p_y,p_z)
        D_prior_black = 0.5 + 1 / np.pi * np.arctan(-(D - 0.3) / 0.1)

        nii_file = nib.Nifti1Image(D_prior_black, affine_matrix)
        save_path = os.path.join(path, 'Prior_P' + str(0) + '.nii')
        nib.save(nii_file, save_path)


        # D_prior = np.where(D_prior >= 0.5, 0, 1)
        D_prior_th_black = np.where((1 - D_prior_black) >= 0.5, 1, 0)
        D_prior_mean_black=np.mean(D_prior_th_black)
        print('mean_black', np.mean(D_prior_th_black))

        connected_regions = find_connected_regions(D_prior_th_black)
        centroids = [calculate_centroid(region) for region in connected_regions]


        num = 0
        Acc = 0
        D_prior_save=np.zeros(D_prior_mean_black.shape)
        D_prior_save_m = np.zeros(D_prior_mean_black.shape)
        for i, centroid in enumerate(centroids):

            if centroid is None:
                pass
            else:
                X,Y,Z=centroid
                # if D_prior_th_black[int(X), int(Y), int(Z)] == 1:
                # print(X,Y,Z)
                D = calculate_geodesic_dist3D(pet_data, int(X),int(Y),int(Z))
                D_prior = 0.5 + 1 / np.pi * np.arctan(-(D - 0.3) / 0.1)
                D_prior_th = np.where(D_prior >= 0.5, 1, 0)
                D_prior_mean = np.mean(D_prior_th)
                # print('mean', D_prior_mean)

                if (D_prior_mean < D_prior_mean_black)|(D_prior_mean<0.1):
                    num+=1
                    print(f"Region {i} centroid: {centroid}")
                    print('mean', D_prior_mean)

                    D_prior_save = np.zeros((144, 144, 144))
                    D_prior_save[2:142, 2:142, 2:142] = D_prior
                    nii_file = nib.Nifti1Image(D_prior_save, affine_matrix)
                    save_path = os.path.join(path, 'Prior_S'+str(num)+'.nii')
                    nib.save(nii_file, save_path)
                    if PET_nii_GT[int(X),int(Y),int(Z)]==1:
                        Acc+=1
                        print('The center point is True')
        if np.count_nonzero(D_prior_save)==0:
            nii_file = nib.Nifti1Image(1-D_prior_black, affine_matrix)
        else:
            if num==1:
                D_prior_save=D_prior_save/D_prior_save.max()
                nii_file = nib.Nifti1Image(D_prior_save, affine_matrix)
            else:
                D_prior_save_m = D_prior_save_m / D_prior_save_m.max()
                nii_file = nib.Nifti1Image(D_prior_save_m, affine_matrix)
        save_path = os.path.join(path, 'D_prior_new.nii')
        nib.save(nii_file, save_path)



        Num_center[f]=num
        Acc_center[f]=Acc
    print(Num_center)
    print(Acc_center)
    mat_file_name=r'D:\Data\hecktor\center_lung_train.mat'
    savemat(mat_file_name, {
        'Num_center': Num_center,
        'Acc_center': Acc_center})


