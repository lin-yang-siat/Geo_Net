from scipy.ndimage import gaussian_filter
# import GeodisTK
import numpy as np
import skfmm
import scipy.ndimage


def calculate_geodesic_dist3D(z, X, Y,Z):
    z = np.array(z)
    z1 = (z - z.mean()) / z.std()
    # print(z1.max())

    beta = 1
    R = np.zeros(np.shape(z))

    R[X, Y, Z] = 1

    z1 = gaussian_filter(z1, sigma=1)
    gx, gy, gz = np.gradient(z1)
    nab_z = np.sqrt(gx ** 2 + gy ** 2+ gz ** 2)

    f = beta * (np.max(z1) - z1) * nab_z ** 2

    f = (1./(f+0.001))
    T = skfmm.travel_time(R-0.5*np.ones(np.shape(R)), speed=f, dx=1.0/np.shape(R)[0], order=1)
    T = T/np.max(T)
    return T




