import numpy as np
import matplotlib.pyplot as plt
from math import floor, cos, sin, acos, asin

def vecmap_dot(x,y):
    return np.einsum('...l,...l->...',x,y)

def clamp(x,inf,sup):
    return np.maximum(inf,np.minimum(sup,x))

def clamp_int(x,inf,sup):
    return clamp(np.round(x),inf,sup)

def normalise(v,axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)

def angle(v1, v2):
    """
    axis must be -1
    """
    dot = vecmap_dot(normalise(v1),normalise(v2))
    if (np.min(dot) < - 1 or np.max(dot) > 1):
        print('Warning in arccos : encountered values out of range. Clipping data to plausible range.', np.min(dot), np.max(dot))
        dot = np.clip(dot, -1, 1)
    aco = np.arccos(dot)
    return aco

def plottable_vector_map(vmap):
    out = vmap - np.min(vmap)
    out /= np.max(out)
    return out

def ready_to_draw(xytab):
    return np.swapaxes(plottable_vector_map(xytab), 0, 1)

def sample_spheres(R,r):
    theta = np.rad2deg(2 * asin(r/R))
    ratio = 360 / theta
    nb_spheres = floor(ratio)
    theta = 360 / nb_spheres
    x = []
    y = []
    for s in range(nb_spheres):
        x.append(R * cos(s*np.deg2rad(theta)))
        y.append(R * sin(s*np.deg2rad(theta)))
    return x,y

def cart2spher(x,y,z):
    rho = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    theta = np.arccos(z / rho)
    phi = np.arctan2(y,x)
    return rho, theta, phi

def spher2cart(rho,theta,phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)

# np.unstack pas dispo en python3.8
# def cross_product_matrix(v):
#     z, u1, u2, u3 = np.broadcast_arrays(*((0,) + np.unstack(v, axis=-1)))
#     result = np.stack([
#         np.stack([z, -u3,  u2], axis=-1),
#         np.stack([ u3, z, -u1], axis=-1),
#         np.stack([-u2,  u1, z], axis=-1)
#     ], axis=-2)
#     return result

def cross_product_matrix(v):
    # On extrait les composantes du vecteur
    u1 = v[..., 0]
    u2 = v[..., 1]
    u3 = v[..., 2]
    z = np.zeros_like(u1)

    # Construction de la matrice antisymétrique
    result = np.stack([
        np.stack([ z, -u3,  u2], axis=-1),
        np.stack([ u3,  z, -u1], axis=-1),
        np.stack([-u2, u1,   z], axis=-1)
    ], axis=-2)
    
    return result


def matrix_to_axis_angle(R):
    trace = np.trace(R, axis1=-2, axis2=-1)
    angle = np.arccos(0.5*(trace-1))
    direction = np.stack([R[...,2,1]-R[...,1,2], R[...,0,2]-R[...,2,0], R[...,1,0]-R[...,0,1]], axis=-1)
    normalization = np.reciprocal(2.0*np.sin(angle))
    axis = np.expand_dims(normalization, axis=-1) * direction
    return axis, angle

def axis_angle_to_matrix(axis, angle):
    K = cross_product_matrix(axis)
    R = np.eye(3) + np.expand_dims(np.sin(angle), axis=(-1,-2)) * K + np.expand_dims(1 - np.cos(angle), axis=(-1,-2)) * (K @ K)
    return R

