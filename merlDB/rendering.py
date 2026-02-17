import numpy as np
import matplotlib.pyplot as plt
from .utils import axis_angle_to_matrix, clamp, normalise, ready_to_draw, sample_spheres, angle
from . import database as db
import random
import sys
import math
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import time
import argparse

sys.path.append("..")

import utilsBRDFDisney

def pixel_viex_dirs_ortho(n,m):
    """Compute the view direction associated with every pixel for orthographic projection

    Args:
        n (int): x resolution 
        m (int): y resolution

    Returns:
        views (array(n,m,3)): A vector map of the view directions
    """    
    views = np.zeros((n,m,3))
    views[:,:,2] = -1
    return views


def render_brdf(mask, view_dirs, normal_map, light_dir, brdf):
    """Deprecated, use render_brdf_vector instead 
    Performs the rendering of an object given a brdf

    Args:
        mask (array(n,m,int)): Binary mask of the object to render
        view_dirs (array(n,m,3)): Vector map of the camera view directions for each pixel
        normal_map (array(n,m,3)): Normal map of the object to render
        light_dir (array(n,m,3)): Vector map of the associated light directions for each pixel
        brdf (func(array(k,3)->array(k,3))): brdf func

    Returns:
        image,phi_d,theta_d,theta_h (array(n,m,3), array(n,m)*3): RGB array
    """        
    n,m,_ = view_dirs.shape
    out = np.zeros((n,m,3))
    ind = np.where(mask)
    sub_v = view_dirs[ind]
    sub_n = normal_map[ind]
    sub_l = light_dir[ind]
    sub_phi_d, sub_theta_d, sub_theta_h  = db.rusinkiewicz_angles(sub_n, sub_v, sub_l)
    out[ind] = brdf(np.stack((sub_phi_d, sub_theta_d, sub_theta_h), axis=-1))

    phi_d = np.zeros((n,m))
    theta_d = np.zeros((n,m))
    theta_h = np.zeros((n,m))
    phi_d[ind] = sub_phi_d
    theta_d[ind] = sub_theta_d
    theta_h[ind] = sub_theta_h
    return out, phi_d, theta_d, theta_h

def render_brdf_vector(view_dirs, normals, light_dirs, brdf):
    """Quicker version of render brdf that uses [k,3] shaped vector lists instead"""
    phi_d, theta_d, theta_h = db.rusinkiewicz_angles(normals, light_dirs, view_dirs)
    # print(np.reshape(np.stack((phi_d, theta_d, theta_h), axis=-1), (-1,3)))
    rendered = brdf(np.reshape(np.stack((phi_d, theta_d, theta_h), axis=-1), (-1,3)))
    # print(rendered.shape)
    return rendered



def sphere_normals(X,Y,r,x0,y0):
    """Compute normals associated with a sphere facing camera

    Args:
        X (array(n,m,3)): x coordinates of the image
        Y (array(n,m,3)): y coordinates of the image
        R (int): radius of the sphere in pixels

    Returns:
        normals,mask (array(n,m,3),array(n,m)): (Sphere normal map, sphere binary mask)
    """    
    n,m = X.shape
    X = X - x0
    Y = Y - y0
    mask = ((np.square(X) + np.square(Y)) < r**2).astype(int)
    SX = X/r * mask
    SY = Y/r * mask
    SZ = -np.sqrt(np.maximum(r**2 - np.square(X) - np.square(Y), 0))/r * mask 

    
    N = np.stack((SX,SY,SZ), axis=-1)
    norms = np.linalg.norm(N,axis=-1) + (1-mask)
    N /= norms[:,:,np.newaxis]

    return mask[:,:,np.newaxis] * N, mask

def simple_sphere_normals(r):
    range = np.linspace(-r,r,2*r)
    X,Y = np.meshgrid(range, range)
    sqX = np.square(X)
    sqY = np.square(Y)
    Z = -np.sqrt(np.maximum(r**2 - sqX -sqY,0))
    mask = (sqX + sqY < r**2).astype(int)
    return mask[:,:,np.newaxis] * normalise(np.stack((X,Y,Z),axis=-1)), mask
        
class Renderer():
    def __init__(self, size, save_path, nb_spheres=1, nb_tours=10, gamma=2.222):
        """Creates a sphere BRDF renderer

        size : resolution of the renderer image
        nb_spheres : The number of spheres that will be rendered on a single image
        nb_tours : The number of circles the light will describe during its total movement 
        """
        self.save_path = save_path
        # Grille de rendu
        self.X, self.Y = np.meshgrid(range(size), range(size))
        self.size = size
        self.gamma = gamma
        self.nb_tours = nb_tours
        # Def des spheres
        self.r = size // (2 * (nb_spheres))
        self.unit_normals, self.unit_mask = simple_sphere_normals(self.r)
        self.unit_ind = np.where(self.unit_mask)
        # Normales et visées
        self.normals, self.views, self.mask = self.__maps()
        self.ind = np.where(self.mask)

    def sub_render(self,i,t,brdf,name='',save=False,grey_levels=False): 
        """Renders a single image of a sphere
        Call render if you wish to render all of the images at once

        i : The number of the image. Useless if not saving
        t : The time caracterizing the light position (0<t<1)
        name : Name of the file to save. Useless if not saving
        save : Tells if the image should be saved
        gray_levels : Compute the grey-level image (is bugged, safer to compute RBG and convert to grey levels afterwards)
        """
        if grey_levels:
            ldim = 1
        else:
            ldim = 3
        l = self.__light(t)
        image = np.zeros((self.size,self.size,ldim))
        ind = self.ind
        rendered = render_brdf_vector(self.views[ind], self.normals[ind], l[ind], brdf)
        pos = rendered > 0
        rendered[pos] = np.power(rendered[pos], 1/self.gamma)
        image[ind] = rendered
        if save:
            to_save = (255*ready_to_draw(image)).astype(np.uint8)
            # print(to_save)
            print(f"{self.save_path + name + str(i) + ".png"}")
            Image.fromarray(to_save).save(self.save_path + name + str(i) + ".png") 
        return image

    def render(self, brdf, nb_images, name='', grey_levels=False, MP=0):
        """Renders a sequence of images
        MP : Set greater to 0 to use multi threading with MP threads
        """
        time = np.linspace(0,1,nb_images)
        
        if MP <= 0:
            for i in tqdm(range(nb_images), "Rendering"):
                self.sub_render(i,time[i],brdf,name,True,grey_levels)
        else:
            with mp.Pool(min(MP, mp.cpu_count() - 1)) as p:
                p.starmap(self.sub_render, [(i,time[i],brdf,name,True,grey_levels) for i in range(nb_images)])



    def __light(self, t):
        t = math.sqrt(t)
        r = t * self.size / 2
        theta = t * 360 * self.nb_tours
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
        z = -math.sqrt(np.maximum(0,(self.size/2)**2 - x**2 - y**2))
        return normalise(np.ones((self.size,self.size,3)) * np.array([x,y,z]))

    def __sphere_centers(self):
        sx = [0]
        sy = [0]
        for R in range(2*self.r, int(self.size/2), 2*self.r):
            nsx, nsy = sample_spheres(R,self.r)
            sx.extend(nsx)
            sy.extend(nsy)
        return sx,sy

    def __maps(self):
        size = self.size
        r = self.r
        sx, sy = self.__sphere_centers()
        normals = np.zeros((size, size, 3))
        views = np.zeros((size,size,3))
        mask = np.zeros((size,size))
        for s in range(len(sx)):
            sxx = int(size//2 + sx[s])
            syy = int(size//2 + sy[s])
            sphereX, sphereY = np.meshgrid(range(sxx-r,sxx+r), range(syy-r,syy+r))
            sphereX = sphereX[self.unit_ind]
            sphereY = sphereY[self.unit_ind]
            normals[sphereX, sphereY] = self.unit_normals[self.unit_ind]
            mask[sphereX, sphereY] = self.unit_mask[self.unit_ind]
            views[sphereX, sphereY] = -normalise(np.array([sx[s], sy[s], math.sqrt((size/2)**2 - sx[s]**2 - sy[s]**2)]))
        ind = np.where(mask)
        v = views[ind]
        world = np.array([0,0,-1]) * np.ones_like(v)
        angles = angle(world,v)
        axes = np.cross(world,v)
        matrices = axis_angle_to_matrix(axes, np.deg2rad(angles))
        normals[ind] = np.einsum('kij,kj->ki',matrices, normals[ind])
        return normals, views, mask
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merldir', default='merlDB/db/brdfs/')
    parser.add_argument('--outdir', default='rendered_frames/')
    parser.add_argument('--mat', default=None)
    args = parser.parse_args()

    renderer = Renderer(size=1000, save_path=args.outdir,nb_spheres=1,nb_tours=10,gamma=2.222)
    # dbuilder = db.DBuilder(interp_method="linear",db_path=args.merldir)
    # if not args.mat is None:
    #     mat = args.mat
    # else:
    #     ldb = dbuilder.list_db()
    #     mat = ldb[random.randint(0,len(ldb)-1)]
    # dbuilder.load_mat(mat)
    # print("Loaded " + mat)
    # brdf = dbuilder.brdf_function(mat)

    brdf = utilsBRDFDisney.brdf_for_rendering
    t1 = time.time()
    renderer.render(brdf, nb_images=10,name="test",MP=4)
    t2 = time.time()
    print("Finished in", t2 -t1, "s")
    
