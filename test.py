import torch
import torch.nn as nn

import numpy as np

from model import EncoderViT3D

folder_brdfs = "brdfs_disney/"

def load_brdf_file(index=2):
    """
    Load BRDF from file located at "{folder_brdfs}/brdf_{index}.npz"
    """

    # Load selected BRDF
    sample = np.load(f"{folder_brdfs}/brdf_{index}.npz", allow_pickle=True)
    brdf = sample["brdf"]
    brdf = brdf / (1.0 + brdf)
    params = sample["params"].item()

    # Load angles (only once)
    angles_file = np.load(f"{folder_brdfs}/angles.npz")
    theta_hs = angles_file["theta_h"]
    theta_ds = angles_file["theta_d"]
    phi_ds = angles_file["phi_d"]

    print(f"Loaded BRDF {index}")
    # print(params)

    return brdf, params, theta_hs, theta_ds, phi_ds

brdf, params, theta_hs, theta_ds, phi_ds = load_brdf_file()

## Convert to torch tensor
rgbs = torch.tensor(brdf)
params = [params['baseColor'][0], params['baseColor'][1], params['baseColor'][2], params['metallic'], params['subsurface'], params['specular'], params['roughness'], params['specularTint'], params['sheen'], params['sheenTint'], params['clearcoat'], params['clearcoatGloss']]
params_disney = torch.tensor(params)
params_disney = params_disney.unsqueeze(0)  # Add batch dimension


rgbs = rgbs.permute(3, 0, 1, 2)   # C D H W
rgbs = rgbs.unsqueeze(0)         # 1 C D H W

## load model for .pt file
model = EncoderViT3D()        # instantiate architecture first
model.load_state_dict(torch.load("results/encoder_disney.pt"))
model.eval()

## predict
loss, pred_params = model(
            rgbs, params_disney
        )

print(loss)
print(pred_params)
print(params_disney)