import torch
import torch.nn as nn

import numpy as np

folder_brdfs = "brdfs_disney/"

def load_brdf_file(index=2):
    """
    Load BRDF from file located at "{folder_brdfs}/brdf_{index}.npz"
    """

    # Load selected BRDF
    sample = np.load(f"{folder_brdfs}/brdf_{index}.npz", allow_pickle=True)
    brdf = sample["brdf"]
    brdf = brdf / (1.0 + brdf)
    params = sample["params"]

    # Load angles (only once)
    angles_file = np.load(f"{folder_brdfs}/angles.npz")
    theta_hs = angles_file["theta_h"]
    theta_ds = angles_file["theta_d"]
    phi_ds = angles_file["phi_d"]

    print(f"Loaded BRDF {index}")
    print(params)

    return brdf, params, theta_hs, theta_ds, phi_ds

brdf, params, theta_hs, theta_ds, phi_ds = load_brdf_file()

## Convert to torch tensor
rgbs = torch.tensor(brdf)
params_disney = torch.tensor(params)

params_disney = torch.repeat_interleave(params_disney, 12, dim=0)

rgbs = torch.einsum('bdhwc->bcdhw', rgbs)

## load model for .pt file
model = torch.jit.load('results/encoder_disney.pt')

## predict
loss, pred_params = model(
            rgbs, params_disney
        )

print(loss)
