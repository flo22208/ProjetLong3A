import os
import numpy as np
from tqdm.asyncio import tqdm
from merlDB import database as db
import jax.numpy as jnp
from render_jax import optimize_params

output_folder = "results/merl_on_jax/"
os.makedirs(output_folder, exist_ok=True)

## Create table of all angles
RES_THETA_H = 90
RES_THETA_D = 90
RES_PHI_D = 180

MAX_THETA_H = 90
MAX_THETA_D = 90
MAX_PHI_D = 180

theta_hs = jnp.deg2rad(jnp.linspace(0, RES_THETA_H, MAX_THETA_H))
theta_ds = jnp.deg2rad(jnp.linspace(0, RES_THETA_D, MAX_THETA_D))
phi_ds = jnp.deg2rad(jnp.linspace(0, RES_PHI_D, MAX_PHI_D))

# Create full meshgrid
TH, TD, PH = jnp.meshgrid(theta_hs, theta_ds, phi_ds, indexing="ij")
angles = jnp.stack([PH, TD, TH], axis=-1).reshape(-1, 3)

## load BRDF from MERL database
dbuilder = db.DBuilder(interp_method="linear",db_path='merlDB/db/brdfs/')
ldb = dbuilder.list_db()
all_material_params = np.array([])

for i in tqdm(range(len(ldb)), desc="Processing materials"):
    mat = ldb[i]
    dbuilder.load_mat(mat)
    # print("Loaded " + mat)
    brdf = dbuilder.brdf_function(mat)
    name = mat

    rgbs = brdf(angles)
    ## Resize to MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3
    brdf_target = rgbs.reshape(MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3)
    brdf_target = jnp.moveaxis(brdf_target, -1, 0)
    brdf_target = brdf_target / (1.0 + brdf_target) # tone mapping

    params, _, loss_finale, loss_hist = optimize_params(TH, TD, PH, brdf_target, steps=1000, lr=1e-2)
    params_full = {
        "baseColor": np.array(params["baseColor"]),
        "metallic": params["metallic"].item(),
        "subsurface": params["subsurface"].item(),
        "specular": params["specular"].item(),
        "roughness": params["roughness"].item(),
        "specularTint" : 0.0,
        "anisotropic" : 0.0,
        "sheen": params["sheen"].item(),
        "sheenTint" : 0.0,
        "clearcoat": params["clearcoat"].item(),
        "clearcoatGloss" : 0.0,
    }
    entry = (name, params_full)

    all_material_params = np.append(all_material_params, entry)
    np.savez(f"{output_folder}/jax.npz", params=all_material_params)
    
data = np.load(f"{output_folder}/jax.npz", allow_pickle=True)
print(data['params'][0])
print(data['params'][1])