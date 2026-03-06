import time
from matplotlib import pyplot as plt
from merlDB import database as db
import jax.numpy as jnp
from render_jax import optimize_params

## load BRDF from MERL database
dbuilder = db.DBuilder(interp_method="linear",db_path='merlDB/db/brdfs/')
ldb = dbuilder.list_db()
mat = "green-metallic-paint"
dbuilder.load_mat(mat)
print("Loaded " + mat)
brdf = dbuilder.brdf_function(mat)
name = mat

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

rgbs = brdf(angles)
## Resize to MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3
brdf_target = rgbs.reshape(MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3)
brdf_target = jnp.moveaxis(brdf_target, -1, 0)
brdf_target = brdf_target / (1.0 + brdf_target) # tone mapping

print("Début de l'optimisation")
t1 = time.time()
params, _, loss_finale, loss_hist, pred_finale = optimize_params(TH, TD, PH, brdf_target, steps=1000, lr=1e-2)
t2 = time.time()
print(f"Finished in {(t2-t1):3f} s")
print(f"loss finale : {loss_finale:6f}")
plt.semilogy(loss_hist)
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("BRDF fitting loss")
plt.show()
params_full = {
    "baseColor": params["baseColor"],
    "metallic": params["metallic"],
    "subsurface": params["subsurface"],
    "specular": params["specular"],
    "roughness": params["roughness"],
    "specularTint" : 0.0,
    "anisotropic" : 0.0,
    "sheen": params["sheen"],
    "sheenTint" : 0.0,
    "clearcoat": params["clearcoat"],
    "clearcoatGloss" : 0.0,
}
print("MERL mean:", brdf_target.mean(axis=(1,2,3)))
print("Disney mean:", pred_finale.mean(axis=(1,2,3)))
pred_finale = jnp.moveaxis(pred_finale,0,-1)
jnp.savez(f"brdfs_disney/brdf_0.npz",
        params=params_full,
        brdf=pred_finale)