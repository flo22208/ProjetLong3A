import numpy as np
import os
from .utils import normalise, angle, vecmap_dot, clamp
import scipy.interpolate as interp
import plotly.graph_objects as go
import sys
import random
import matplotlib.pyplot as plt
import argparse
import tqdm


db_path = "./db/brdfs/" 
RES_THETA_H = 90
RES_THETA_D = 90
RES_PHI_D = 180

RED_SCALE = 1/1500
GREEN_SCALE = 1.15/1500
BLUE_SCALE = 1.66/1500

LABELS_MAT = ['other', 'alum', 'fabric', 'obsidian', 'steel', 'phenolic', 'plastic', 'acrylic', 'paint', 'rubber', 'latex', 'two-layer']
LABELS_COLOR = ['other', 'beige', 'black', 'blue', 'chrome', 'changing','gold', 'gray', 'green', 'brown', 'maroon', 'orange', 'pearl', 'pink', 'purple', 'red', 'silver', 'white', 'violet', 'yellow']

def find_label(labels_to_use, mat):
    label = 0
    for i in range(len(labels_to_use)):
        if mat.find(labels_to_use[i]) != -1:
            label = i
            break 
    return label


def rusinkiewicz_angles(n, wi, wo, clamp_values=True):
    """Compute Rusinkiewicz angles from classic system vectors

    Args:
        n (array(k,3)): Normals
        wi (array(k,3)): Light directions
        wo (array(k,3)): Camera directions

    Returns:
        angles (array(k,3)): phi_d, theta_d, theta_h for each input
    """    

    k = n.shape[0]

    phi_d = np.zeros(k)
    theta_d = np.zeros(k)
    theta_h = np.zeros(k)

    h = normalise(wi + wo)
    mask = np.where(np.sum(np.square(h-n), axis=1) > 0)
    v1 = normalise(np.cross(n[mask,:],h[mask,:], axis=-1))
    v2 = np.cross(v1,h[mask,:], axis=-1)

    phi_d[mask] = (np.arctan2(vecmap_dot(wi[mask,:], v1), vecmap_dot(wi[mask,:], v2))) % np.pi
    theta_d[mask] = angle(h[mask,:], wi[mask,:])
    theta_h[mask] = angle(h[mask,:], n[mask,:])

    if clamp_values:
        return clamp(phi_d, 0, np.pi*(RES_PHI_D-1)/RES_PHI_D), clamp(theta_d, 0, (np.pi/2) *(RES_THETA_D-1)/RES_THETA_H), clamp(theta_h, 0, (np.pi/2) * (RES_THETA_H-1)/RES_THETA_H) 
    else:
        return phi_d, theta_d, theta_h
    



def rusinkiewicz_angles_old(n, wi, wo, clamp_values=True):
    """Compute Rusinkiewicz angles from classic system vectors

    Args:
        n (array(k,3)): Normals
        wi (array(k,3)): Light directions
        wo (array(k,3)): Camera directions

    Returns:
        angles (array(k,3)): phi_d, theta_d, theta_h for each input
    """    
    h = normalise(wi + wo)
    v1 = normalise(np.cross(n,h, axis=-1))
    v2 = np.cross(v1,h, axis=-1)

    phi_d = (np.arctan2(vecmap_dot(wi, v1), vecmap_dot(wi, v2))) % np.pi
    theta_d = angle(h, wi)
    theta_h = angle(h, n)

    if clamp_values:
        return clamp(phi_d, 0, np.pi*(RES_PHI_D-1)/RES_PHI_D), clamp(theta_d, 0, (np.pi/2) *(RES_THETA_D-1)/RES_THETA_H), clamp(theta_h, 0, (np.pi/2) * (RES_THETA_H-1)/RES_THETA_H) 
    else:
        return phi_d, theta_d, theta_h
    


def classic_angles(n, wi, wo):
    
    t = np.array([0,1,0]) * np.ones_like(n)
    v2 = np.cross(n, t, axis=-1)
    t = np.cross(v2, n, axis=-1)
    phi_i = np.rad2deg(np.arctan2(vecmap_dot(wi, v2), vecmap_dot(wi, t))) 
    phi_o = np.rad2deg(np.arctan2(vecmap_dot(wo, v2), vecmap_dot(wo, t))) 
    phi_imo = (phi_i - phi_o)
    theta_i = angle(n, wi)
    theta_o = angle(n, wo)

    return phi_imo, theta_i, theta_o





def readbin(filename):
        with open(filename) as f:
            dims = np.fromfile(f, dtype=np.int32, count=3)
            n = np.prod(dims)
            values = np.fromfile(f, dtype=np.double, count=3*n)
            values = np.reshape(values, (RES_PHI_D, RES_THETA_D, RES_THETA_H, -1), 'F') * np.asarray([RED_SCALE, GREEN_SCALE, BLUE_SCALE])
        return values


def writebin(filename, values):
    values /= np.asarray([RED_SCALE, GREEN_SCALE, BLUE_SCALE])
    to_write = np.reshape(values, -1, order='F')
    dims = np.int32([RES_PHI_D, RES_THETA_D, RES_THETA_H])
    with open(filename, 'wb') as f:
        dims.tofile(f)
        to_write.astype(np.double).tofile(f)



class DBuilder():
    """Database loader
    """    
    def __init__(self, interp_method="linear", db_path: str = db_path, albedo_mapping=None, file_extension=".binary"):
        """Initiate database in a folder

        Args:
            interp_method (str, optional): Interpolation method to use (see ReguliarGridInterpolator), or None
            db_path (str, optional): The path of the folder containing the binary brdf file. Defaults to db_path.
        """        
        self.interp_method = interp_method
        self.file_extension = file_extension
        self.db_path = db_path
        self.mats = []
        self.table = {}
        self.interpolators = {}
        self.albedo_mapping = albedo_mapping
        self.__scan_db()

    def mat_path(self, mat: str):
        """Get the path associated with the name of a material

        Args:
            mat (str): Material name

        Returns:
            str: full material path
        """        
        return self.db_path + mat + self.file_extension
    
    def __scan_db(self):
        self.mats = os.listdir(self.db_path)
        self.mats = [f[:-len(self.file_extension)] for f in self.mats]

    def list_db(self):
        """List all accessible materials

        Returns:
            mats list(str): all mats
        """        
        return self.mats
    
    def list_db_loaded(self):
        """List all loaded materials

        Returns:
            mats list(str): all mats
        """
        return list(self.table.keys())


    def load_mat(self, mat: str):
        """Load a material. Will crash if material is not accessible.

        Args:
            mat (str): Material to load
        """
        if mat in self.table.keys(): return
        filename = self.mat_path(mat)
        values = readbin(filename)
        if self.albedo_mapping != None:
            values = self.albedo_mapping(values)

        self.compute_interp(mat, values)

    def load_all(self):
        """Loads all materials available
        """
        for mat in self.list_db():
            self.load_mat(mat)

    def compute_interp(self, mat, values):
        """Computes the RegularGridInterpolator from the BRDF grids
        """
        phi_d = (np.linspace(0, 1, RES_PHI_D, endpoint=False)) * np.pi
        theta_d = (np.linspace(0, 1, RES_THETA_D, endpoint=False)) * np.pi/2
        theta_h = (np.power(np.linspace(0, 1, RES_THETA_H, endpoint=False),2)) * np.pi/2
        grid = (phi_d, theta_d, theta_h)
        
        
        self.table[mat] = values
        if self.interp_method != None:
            interpolator = interp.RegularGridInterpolator(grid, values, method=self.interp_method, fill_value=[0,0,0], bounds_error=False)
            self.interpolators[mat] = interpolator

    def plot_histogram(self, save_path, sample_res=30, value_res=1):
        """Computes the histogrames of the BRDF values across all loaded data"""
        phi_d = np.deg2rad(np.linspace(0, RES_PHI_D, sample_res))
        theta_d = np.deg2rad(np.linspace(0, RES_THETA_D, sample_res))
        theta_h = np.deg2rad(np.linspace(0, RES_THETA_H, sample_res))
        phi_d, theta_d, theta_h = np.meshgrid(phi_d, theta_d, theta_h)
        phi_d = phi_d.flatten()
        theta_d = theta_d.flatten()
        theta_h = theta_h.flatten()
        hist = {}
        for mat in self.list_db_loaded():
            brdf = self.brdf_function(mat)
            colors = brdf(np.stack((phi_d, theta_d, theta_h), axis=-1))
            gray = np.mean(colors, axis=-1)
            values = np.unique(value_res * gray.astype(int))
            for value in values:
                if value in hist:
                    hist[value] += 1
                else:
                    hist[value] = 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(np.array(list(hist.keys()))/value_res, np.array(list(hist.values()))/value_res)
        fig.savefig(save_path + 'hist.png')
        return hist


    def brdf_data(self, mat: str):
        """brdf data of given material

        Args:
            mat (str): _description_

        Raises:
            BaseException: Material does not exist or was not loaded 

        Returns:
            brdf (array(180,90,90,3)): brdf data
        """        
        if (mat in self.table.keys()):
            return self.table[mat]
        else:
            raise BaseException(mat + " does not exist or was not loaded")

    def brdf_function(self, mat: str):
        """brdf func of given material

        Args:
            mat (str): _description_

        Raises:
            BaseException: Material does not exist or was not loaded, or dbuilder interp is set to none

        Returns:
            brdf (func(array(k,3)->array(k,3))): brdf function
        """        
        if (mat in self.table.keys()):
            return self.interpolators[mat]
        else:
            raise BaseException(mat + " does not exist or was not loaded")
        



    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merldir', default='merlDB/db/brdfs/')
    parser.add_argument('--mat', default=None)
    parser.add_argument('--alpha', default=0.5)
    args = parser.parse_args()

    dbuilder = DBuilder(db_path=args.merldir)
    ldb = dbuilder.list_db()
    mat = ldb[random.randint(0, len(ldb)-1)]
    if not args.mat is None: mat = args.mat
    print("Plotting " + mat + " brdf")
    dbuilder.load_mat(mat)
    brdf = dbuilder.brdf_function(mat)

    res = 20
    alpha = float(args.alpha)
    
    phi_d = np.deg2rad(np.linspace(0, RES_PHI_D, res))
    theta_d = np.deg2rad(np.linspace(0, RES_THETA_D, res))
    theta_h = np.deg2rad(np.linspace(0, RES_THETA_H, res))
    phi_d, theta_d, theta_h = np.meshgrid(phi_d, theta_d, theta_h)
    phi_d = phi_d.flatten()
    theta_d = theta_d.flatten()
    theta_h = theta_h.flatten()
    colors = brdf(np.stack((phi_d, theta_d, theta_h), axis=-1))
    colors = np.hstack((colors, alpha*np.ones((colors.shape[0],1))))

    fig = go.Figure(data=[go.Scatter3d(x=phi_d, y=theta_d, z=theta_h, mode='markers', 
                                       marker=dict(size=5, color=colors))])
    fig.show()


    


    


