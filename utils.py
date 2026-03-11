import numpy as np
from numpyBRDF import BRDF

folder_brdfs = "brdfs_disney/"

# ==========================
# Other helpers
# ==========================

def orthogonal_vector(v):
    """
    Returns a 3D vector orthogonal to input v.
    Uses cross product with a safe basis vector.
    """
    if np.allclose(v, 0):
        raise ValueError("Cannot find orthogonal vector to zero vector")
    # Index of largest abs component
    i = np.argmax(np.abs(v))
    # Basis vector e_i
    e = np.ones(3)
    e[i] = - (v[(i+1) % len(v)] + v[(i+2) % len(v)]) / v[i]
    # Cross product gives orthogonal vector
    w = np.cross(v, e)
    return w

brdf = None
params = None
theta_hs = None
theta_ds = None
phi_ds = None


def load_brdf_file(index=2):
    """
    Load BRDF from file located at "{folder_brdfs}/brdf_{index}.npz"
    """
    global brdf, params, theta_hs, theta_ds, phi_ds

    # Load selected BRDF
    sample = np.load(f"{folder_brdfs}/brdf_{index}.npz", allow_pickle=True)
    brdf = sample["brdf"]
    # brdf = brdf / (1.0 + brdf) tonemapping has to be already done in the file in some way
    params = sample["params"]

    # Load angles (only once)
    angles_file = np.load(f"{folder_brdfs}/angles.npz")
    theta_hs = angles_file["theta_h"]
    theta_ds = angles_file["theta_d"]
    phi_ds = angles_file["phi_d"]

    print(f"Loaded BRDF {index}")
    print(params)


def brdf_for_rendering(angles):
    """
    Function that fetches BRDF values in the "brdf" variable for all angles in angles
    angles : size (N,3)
    """

    res = np.zeros(angles.shape)

    for i, triplet in enumerate(angles):
        phi_d, theta_d, theta_h = triplet

        # Find closest triplet in theta_hs, theta_ds and phi_ds
        phi_d_idx = np.argmin(np.abs(phi_ds - phi_d))
        theta_d_idx = np.argmin(np.abs(theta_ds - theta_d))
        theta_h_idx = np.argmin(np.abs(theta_hs - theta_h))

        # Get the corresponding RGB value
        rgb = np.array(brdf[theta_h_idx, theta_d_idx, phi_d_idx])
        res[i] = rgb

    return res

def brdf_for_rendering_vec(light_dirs, view_dirs, normals):
    """
    Function that computes BRDF for all light_dirs, view_dirs, normals
    """

    res = np.zeros((light_dirs.shape))

    for i in range(len(light_dirs)):
        L,V,N = light_dirs[i], view_dirs[i], normals[i]
        N = N / np.linalg.norm(N)
        X = orthogonal_vector(N)
        X = X / np.linalg.norm(X)
        Y = np.cross(X,N)
        rgb = BRDF(L,V,N,X,Y)
        rgb = rgb / (1.0 + rgb)

        res[i] = rgb

    return res

if __name__ == "__main__":
    pass