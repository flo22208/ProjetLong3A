import matplotlib.pyplot as plt
import time
import jax
import optax
import numpy as np
import jax.numpy as jnp
import numpyBRDF
from BRDFSaver import MAX_PHI_D, MAX_THETA_D, MAX_THETA_H, RES_PHI_D, RES_THETA_D, RES_THETA_H
import jaxBRDF

BRDF_jitted = jax.jit(jaxBRDF.BRDF_jax)

def init_param():
    scale = 0.5    
    material_params = {
        "baseColor": jnp.array(scale * np.random.randn(3)),
        "metallic": jnp.array(scale * np.random.randn()),
        "subsurface": jnp.array(scale * np.random.randn()),
        "specular": jnp.array(scale * np.random.randn()),
        "roughness": jnp.array(scale * np.random.randn()),
        "sheen": jnp.array(scale * np.random.randn()),
        "clearcoat": jnp.array(scale * np.random.randn()),
    }
    return material_params
    
def loss_fn(params, L, V, N, X, Y, brdf_target):
    baseColor = jax.nn.sigmoid(params["baseColor"])
    metallic  = jax.nn.sigmoid(params["metallic"])
    specular  = jax.nn.sigmoid(params["specular"])
    roughness = jax.nn.sigmoid(params["roughness"])
    sheen     = jax.nn.sigmoid(params["sheen"])
    clearcoat = jax.nn.sigmoid(params["clearcoat"])
    subsurface = jax.nn.sigmoid(params["subsurface"])
    pred = BRDF_jitted(L, V, N, X, Y, baseColor=baseColor,
    metallic=metallic,
    subsurface=subsurface,
    specular=specular,
    specularTint=0.0,
    roughness=roughness,
    sheen=sheen,
    sheenTint=0,
    clearcoat=clearcoat,
    clearcoatGloss=0)
    pred = pred / (1.0 + pred) # tone mapping
    # eps = 1e-6
    # loss = jnp.mean((pred - brdf_target)**2) # MSE 2)
    # log_term = (jnp.log(pred+eps) - jnp.log(brdf_target+eps))**2
    # lin_term = (pred - brdf_target)**2
    # loss = jnp.mean(0.7 * log_term + 0.3 * lin_term) # Log-MSE + MSE 4)
    # loss = jnp.mean(((pred - brdf_target)**2) / (brdf_target**2 + eps)) # MSE relative 6)
    # loss = jnp.mean((jnp.log(pred+eps) - jnp.log(brdf_target+eps))**2) # Log-MSE 5)
    loss = optax.huber_loss(pred, brdf_target, delta=0.01).mean() # Huber Loss 1)
    # loss = optax.log_cosh(pred, brdf_target).mean() # Log-Cos 3)
    return loss    

def optimize_params(TH, TD, PH, brdf_target, steps=1000, lr=1e-2): # 2000 5e-3
    params = init_param()
    params_init_jax = params
    L, V, N, X, Y = jaxBRDF.rusinkiewicz_to_LV_jax(TH, 0.0, TD, PH)
    # 1) Créer l’optimiseur
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # 2) Fonction step
    @jax.jit
    def step(params, opt_state,L, V, N, X, Y, brdf_target):
        loss, grads = jax.value_and_grad(loss_fn)(params, L, V, N, X, Y, brdf_target)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    loss_hist = []
    # 3) Boucle d’optimisation
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state,L, V, N, X, Y, brdf_target)
        loss_hist.append(loss)
        
    params = {k: jax.nn.sigmoid(v) for k, v in params.items()}
    pred_finale = BRDF_jitted(L, V, N, X, Y, baseColor=params["baseColor"],
    metallic=params["metallic"],
    subsurface=params["subsurface"],
    specular=params["specular"],
    specularTint=0.0,
    roughness=params["roughness"],
    sheen=params["sheen"],
    sheenTint=0,
    clearcoat=params["clearcoat"],
    clearcoatGloss=0)

    return params, params_init_jax, loss, loss_hist, pred_finale

def test():
    print("Test égalité Jax-Numpy")
    baseColor = jnp.array(np.random.rand(3))
    metallic = jnp.array(np.random.rand())
    subsurface = jnp.array(np.random.rand())
    specular = jnp.array(np.random.rand())
    roughness = jnp.array(np.random.rand())
    specularTint = jnp.array(np.random.rand())
    anisotropic = jnp.array(0.0)
    sheen = jnp.array(np.random.rand())
    sheenTint = jnp.array(np.random.rand())
    clearcoat = jnp.array(np.random.rand())
    clearcoatGloss = jnp.array(np.random.rand())
    theta_hs = np.deg2rad(np.linspace(0, RES_THETA_H, MAX_THETA_H))
    theta_ds = np.deg2rad(np.linspace(0, RES_THETA_D, MAX_THETA_D))
    phi_ds = np.deg2rad(np.linspace(0, RES_PHI_D, MAX_PHI_D))
    TH, TD, PH = jnp.meshgrid(theta_hs, theta_ds, phi_ds, indexing="ij")
    L, V, N, X, Y = jaxBRDF.rusinkiewicz_to_LV_jax(TH, 0.0, TD, PH)
    out_jax = jaxBRDF.BRDF_jax(L, V, N, X, Y, baseColor,metallic,subsurface,specular,
    roughness,specularTint,anisotropic,
    sheen,sheenTint,clearcoat,clearcoatGloss)
    brdf = np.zeros((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3), dtype=np.float32)
    for hi in range(MAX_THETA_H):
        for di in range(MAX_THETA_D):
            for pi in range(MAX_PHI_D):
               
                vals = numpyBRDF.BRDF(
                    L[...,hi,di,pi], V[...,hi,di,pi], N[...,hi,di,pi], X[...,hi,di,pi], Y[...,hi,di,pi],
                    baseColor, metallic, subsurface,
                    specular, roughness, specularTint,
                    anisotropic, sheen, sheenTint,
                    clearcoat, clearcoatGloss
                )
                brdf[hi, di, pi] = vals
    brdf = np.moveaxis(brdf,-1,0)
    print(f"brdf numpy shape : {brdf.shape}")
    print(f"brdf jax shape : {out_jax.shape}")
    abs_err = jnp.abs(brdf - out_jax)
    rel_err = abs_err / (jnp.abs(brdf) + 1e-8)
    print(f"Erreur absolue : {np.mean(abs_err)}")
    print(f"Erreur relative : {np.mean(rel_err)}")
    
def affiche(x):
    x = np.array(x)
    if x.ndim == 0:
        return f"{float(x):.4f}"
    else:
        return "[" + ", ".join(f"{float(v):.4f}" for v in x) + "]"

if __name__ == "__main__":
    # test()
    data = np.load("brdfs_disney/brdf_5.npz", allow_pickle=True)
    brdf_target = jnp.array(data["brdf"])  # (H,D,P,3)
    params_init = data["params"].item()
    brdf_target = jnp.moveaxis(brdf_target,-1,0)
    brdf_target = brdf_target / (1.0 + brdf_target) # tone mapping
    angles = np.load("brdfs_disney/angles.npz")
    theta_hs = jnp.array(angles["theta_h"])
    theta_ds = jnp.array(angles["theta_d"])
    phi_ds = jnp.array(angles["phi_d"])

    # Create full meshgrid
    TH, TD, PH = jnp.meshgrid(theta_hs, theta_ds, phi_ds, indexing="ij")
    t1 = time.time()
    params, params_init_jax, loss_finale, loss_hist, pred_finale = optimize_params(TH, TD, PH, brdf_target, steps=1000, lr=5e-2)
    t2 = time.time()
    print(f"Finished in {(t2-t1):3f} s")
    print("\n=== Parameters comparison ===")
    abs_params = {}
    for key in params.keys():
        original = params_init[key]
        prediction = params[key]
        abs_params[key] = np.mean(np.abs(original - prediction))
        print(f"{key:12s} | Original: {affiche(original)}   →   Prédiction: {affiche(prediction)}")
    plt.semilogy(loss_hist)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("BRDF fitting loss")
    plt.show()
    np.savez(f"brdfs_disney/brdf_pred.npz",
             params=params,
             brdf=pred_finale)
    abs_global = np.mean([v for v in abs_params.values()])
    print(f"\nDifférence absolue globale paramètres : {abs_global:.4f}")
