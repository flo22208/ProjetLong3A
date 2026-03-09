# Reads Disney representations of all BRDFs in MERL DB and computes a loss for each material

import numpy as np
import merlDB.database as db
from torchBRDF import BRDF, rusinkiewicz_to_LV
import torch

file = "results/merl_on_jax/jax.npz"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read file
data = np.load(file, allow_pickle=True)

# load MERL db
dbuilder = db.DBuilder(interp_method="linear",db_path='merlDB/db/brdfs/')
ldb = dbuilder.list_db()

RES_THETA_H = 90
RES_THETA_D = 90
RES_PHI_D = 180

MAX_THETA_H = 90
MAX_THETA_D = 90
MAX_PHI_D = 180

theta_hs = np.deg2rad(np.linspace(0, RES_THETA_H, MAX_THETA_H))
theta_ds = np.deg2rad(np.linspace(0, RES_THETA_D, MAX_THETA_D))
phi_ds = np.deg2rad(np.linspace(0, RES_PHI_D, MAX_PHI_D))

angles = np.zeros((MAX_THETA_H*MAX_THETA_D*MAX_PHI_D, 3))

# fill angles with all tripelts of theta_h, theta_d, phi_d
idx = 0
for hi in range(MAX_THETA_H):
    for di in range(MAX_THETA_D):
        for pi in range(MAX_PHI_D):
            angles[idx] = [phi_ds[pi], theta_ds[di], theta_hs[hi]]
            idx += 1

all_losses = {}

for i in range(len(data["params"])//2):
    mat = data["params"][2*i]
    params = data["params"][2*i+1]
    
    dbuilder.load_mat(mat)
    print("Loaded " + mat)
    brdf_function = dbuilder.brdf_function(mat)

    gt_brdf = brdf_function(angles)
    gt_brdf = gt_brdf / (1.0 + gt_brdf)  # tonemapping

    gt_brdf = gt_brdf.reshape((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3))

    wi, wo, N = rusinkiewicz_to_LV(device)

    # Convert the parameters to the format expected by the BRDF function
    params_disney_list = [[params["baseColor"][0], params["baseColor"][1], params["baseColor"][2], params["metallic"], params["subsurface"], params["specular"], params["roughness"], params["specularTint"], params["sheen"], params["sheenTint"], params["clearcoat"], params["clearcoatGloss"] ]]
    params_disney = torch.tensor(params_disney_list, device=device, dtype=torch.float32)

    rgbs = BRDF(params_disney, wi, wo, N)
    # remove batch dim
    pred_brdf = rgbs.squeeze(0) # D H W C
    mask = torch.isinf(pred_brdf)
    pred_brdf = pred_brdf / (1 + pred_brdf)
    pred_brdf[mask] = 1.0
    pred_brdf = pred_brdf.cpu().numpy()

    # MSE
    loss = np.mean((gt_brdf - pred_brdf)**2)
    print(loss)
    
    all_losses[mat] = loss

print(all_losses)
print("Average loss: ", np.mean(list(all_losses.values())))
    