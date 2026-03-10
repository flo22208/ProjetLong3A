from tqdm.asyncio import tqdm

from render_jax import optimize_params
import jax.numpy as jnp
import jax
import jaxBRDF
import time
import numpy as np
from BRDFSaver import MAX_PHI_D, MAX_THETA_D, MAX_THETA_H, RES_PHI_D, RES_THETA_D, RES_THETA_H
import matplotlib.pyplot as plt

BRDF_jitted = jax.jit(jaxBRDF.BRDF_jax)

RES_THETA_H = 90
RES_THETA_D = 90
RES_PHI_D = 180

MAX_THETA_H = 90
MAX_THETA_D = 90
MAX_PHI_D = 180

theta_hs = jnp.deg2rad(jnp.linspace(0, RES_THETA_H, MAX_THETA_H))
theta_ds = jnp.deg2rad(jnp.linspace(0, RES_THETA_D, MAX_THETA_D))
phi_ds = jnp.deg2rad(jnp.linspace(0, RES_PHI_D, MAX_PHI_D))

TH, TD, PH = jnp.meshgrid(theta_hs, theta_ds, phi_ds, indexing="ij")
L, V, N, X, Y = jaxBRDF.rusinkiewicz_to_LV_jax(TH, 0.0, TD, PH)

all_losses = []
all_abs_diff_per_params = jnp.array([])
N_tries = 50
for i in tqdm(range(N_tries), desc="Processing materials"):
    material_params = {
        "baseColor": jnp.array(np.random.rand(3)),
        "metallic": jnp.array( np.random.rand()),
        "subsurface": jnp.array( np.random.rand()),
        "specular": jnp.array( np.random.rand()),
        "roughness": jnp.array( np.random.rand()),
        # "specularTint": jnp.array( np.random.rand()),
        # "anisotropic": jnp.array( np.random.rand()),
        "sheen": jnp.array( np.random.rand()),
        # "sheenTint": jnp.array( np.random.rand()),
        "clearcoat": jnp.array( np.random.rand()),
        # "clearcoatGloss": jnp.array( np.random.rand()),
    }

    baseColor = material_params["baseColor"]
    metallic  = material_params["metallic"]
    subsurface = material_params["subsurface"]
    specular  = material_params["specular"]
    roughness = material_params["roughness"]
    specularTint = 0.0 # material_params["specularTint"]
    anisotropic = 0.0 # material_params["anisotropic"]
    sheen     = material_params["sheen"]
    sheenTint = 0.0 # material_params["sheenTint"]
    clearcoat = material_params["clearcoat"]
    clearcoatGloss = 0.0 # material_params["clearcoatGloss"]

    brdf_target = BRDF_jitted(L, V, N, X, Y, baseColor=baseColor,
    metallic=metallic,
    subsurface=subsurface,
    specular=specular,
    specularTint=specularTint,
    roughness=roughness,
    sheen=sheen,
    sheenTint=sheenTint,
    clearcoat=clearcoat,
    clearcoatGloss=clearcoatGloss)
    brdf_target = brdf_target / (1.0 + brdf_target)

    params, params_init_jax, loss_finale, loss_hist = optimize_params(TH, TD, PH, brdf_target, steps=1000, lr=1e-2)
    
    # detect is there is nan loss and if yes retry optimization with a different initialization
    while jnp.isnan(loss_finale):
        print(f"Loss is nan for try {i}, retrying with different initialization")
        params, params_init_jax, loss_finale, loss_hist = optimize_params(TH, TD, PH, brdf_target, steps=1000, lr=1e-2)

    all_losses.append(loss_finale.item())
    temp_abs_diff_per_param = jnp.abs(material_params["baseColor"] - params["baseColor"])
    temp_abs_diff_per_param = jnp.concatenate([temp_abs_diff_per_param, jnp.abs(material_params["metallic"] - params["metallic"]).reshape(1),
    jnp.abs(material_params["subsurface"] - params["subsurface"]).reshape(1),
    jnp.abs(material_params["specular"] - params["specular"]).reshape(1),
    jnp.abs(material_params["roughness"] - params["roughness"]).reshape(1),
    # jnp.abs(material_params["specularTint"] - params["specularTint"]).reshape(1),
    # jnp.abs(material_params["anisotropic"] - params["anisotropic"]).reshape(
    jnp.abs(material_params["sheen"] - params["sheen"]).reshape(1),
    # jnp.abs(material_params["sheenTint"] - params["sheenTint"]).reshape(1),
    jnp.abs(material_params["clearcoat"] - params["clearcoat"]).reshape(1),
    # jnp.abs(material_params["clearcoatGloss"] - params["clearcoatGloss"]).reshape(1)
    ])
    if len(all_abs_diff_per_params) == 0:
        all_abs_diff_per_params = temp_abs_diff_per_param
    else:
        all_abs_diff_per_params = jnp.vstack([all_abs_diff_per_params, temp_abs_diff_per_param])
    
    

print(f"Average loss on BRDFs over {N_tries} tries: {(sum(all_losses)/N_tries):.6f}")

print(f"Average absolute difference per Disney parameter over {N_tries} tries: {jnp.mean(all_abs_diff_per_params, axis=0)}")

print(f"Standard deviation of absolute difference per Disney parameter over {N_tries} tries: {jnp.std(all_abs_diff_per_params, axis=0)}")

print(f"Min and max absolute difference per Disney parameter over {N_tries} tries: {jnp.min(all_abs_diff_per_params, axis=0)}, {jnp.max(all_abs_diff_per_params, axis=0)}")