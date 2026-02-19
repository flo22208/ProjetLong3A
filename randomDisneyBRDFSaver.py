import numpy as np
import random
import utilsBRDFDisney
import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

folder_brdfs = "brdfs_disney/"

RES_THETA_H = 90
RES_THETA_D = 90
RES_PHI_D = 180

MAX_THETA_H = 90
MAX_THETA_D = 90
MAX_PHI_D = 180

theta_hs = np.deg2rad(np.linspace(0, RES_THETA_H, MAX_THETA_H))
theta_ds = np.deg2rad(np.linspace(0, RES_THETA_D, MAX_THETA_D))
phi_ds = np.deg2rad(np.linspace(0, RES_PHI_D, MAX_PHI_D))

# Save once
np.savez(f"{folder_brdfs}/angles.npz", theta_h=theta_hs, theta_d=theta_ds, phi_d=phi_ds)


def generate_brdf(i):
    # Make randomness process-safe
    rng = np.random.default_rng(seed=i)

    baseColor = rng.random(3)
    metallic = rng.random()
    subsurface = rng.random()
    specular = rng.random()
    roughness = rng.random()
    specularTint = rng.random()
    anisotropic = 0.0
    sheen = rng.random()
    sheenTint = rng.random()
    clearcoat = rng.random()
    clearcoatGloss = rng.random()

    material_params = {
        "baseColor": baseColor,
        "metallic": metallic,
        "subsurface": subsurface,
        "specular": specular,
        "roughness": roughness,
        "specularTint": specularTint,
        "anisotropic": anisotropic,
        "sheen": sheen,
        "sheenTint": sheenTint,
        "clearcoat": clearcoat,
        "clearcoatGloss": clearcoatGloss,
    }

    brdf = np.zeros((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3), dtype=np.float32)

    for hi in range(MAX_THETA_H):
        for di in range(MAX_THETA_D):
            for pi in range(MAX_PHI_D):
                theta_h = theta_hs[hi]
                theta_d = theta_ds[di]
                phi_d = phi_ds[pi]

                L, V, N_vec, X, Y = utilsBRDFDisney.rusinkiewicz_to_LV(
                    theta_h, theta_d, phi_d
                )

                vals = utilsBRDFDisney.BRDF(
                    L, V, N_vec, X, Y,
                    baseColor, metallic, subsurface,
                    specular, roughness, specularTint,
                    anisotropic, sheen, sheenTint,
                    clearcoat, clearcoatGloss
                )
                if vals[0] != -1:
                    brdf[hi, di, pi] = vals

    np.savez(f"{folder_brdfs}/brdf_{i}.npz",
             params=material_params,
             brdf=brdf)

    return i

if __name__ == "__main__":
    N = 10
    os.makedirs(folder_brdfs, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(generate_brdf, range(N)), total=N))

    # --- Load angles ---
    angles = np.load(f"{folder_brdfs}/angles.npz")
    theta_hs = angles["theta_h"]
    theta_ds = angles["theta_d"]
    phi_ds = angles["phi_d"]

    # Create full meshgrid
    TH, TD, PH = np.meshgrid(theta_hs, theta_ds, phi_ds, indexing="ij")

    # --- Load one BRDF sample ---
    sample = np.load(f"{folder_brdfs}/brdf_0.npz", allow_pickle=True)
    params = sample["params"].item()  # material parameters
    brdf = sample["brdf"]  # shape (RES_THETA_H, RES_THETA_D, RES_PHI_D, 3)

    # --- Map angles to RGB values ---
    # Flatten everything for easy pairing
    theta_h_flat = TH.ravel()
    theta_d_flat = TD.ravel()
    phi_d_flat = PH.ravel()
    brdf_flat = brdf.reshape(-1, 3)  # flatten last 3D grid to 2D (N_angles x 3)

    # Now we can access the correspondence as tuples
    angle_to_rgb = list(zip(theta_h_flat, theta_d_flat, phi_d_flat, brdf_flat))

    # Example: print first 5 entries
    for entry in angle_to_rgb[300:350]:
        th, td, pd, rgb = entry
        # print(f"Theta_h={th:.3f}, Theta_d={td:.3f}, Phi_d={pd:.3f} => RGB={rgb}")

