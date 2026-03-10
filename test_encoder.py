import torch
import torch.nn as nn

import numpy as np

from model import EncoderViT3D

from torchBRDF import BRDF, rusinkiewicz_to_LV

folder_brdfs = "brdfs_disney/"

epochs = 3000
embed_dim = 96
latent_space_dim = 8
where_zeros = [4, 7, 9, 11]
# where_zeros = [7, 11]
# where_zeros = []
batch_size = 20

model_file = f"results/encoder_disney_{embed_dim}_{latent_space_dim}_{epochs}_{batch_size}_0.0088.pt"

## load model for .pt file
model = EncoderViT3D(embed_dim=embed_dim, latent_space_dim=latent_space_dim)        # instantiate architecture first
model.load_state_dict(torch.load(model_file))
model.to('cuda')
model.eval()

all_losses = []
all_abs_diff_per_params = []
N_tries = 500
for i in range(N_tries):
    params_disney = torch.rand(batch_size, 12, device='cuda')
    for i in where_zeros:
            params_disney[:, i:i+1] = 0.0
    model_params_disney = torch.zeros(batch_size, latent_space_dim, device='cuda')
    cpt = 0
    for i in range(12):
        if not(params_disney[: , i:i+1].sum() == 0):
            model_params_disney[:, cpt:cpt+1] = params_disney[:, i:i+1]
            cpt +=1
    
    wi, wo, N = rusinkiewicz_to_LV('cuda')
    rgbs = BRDF(params_disney, wi, wo, N)

    # Tonemapping
    mask = torch.isinf(rgbs)
    rgbs = rgbs / (1 + rgbs)
    rgbs[mask] = 1.0
    rgbs = torch.einsum('bdhwc->bcdhw', rgbs)

    ## predict
    loss, pred_params = model(
                rgbs, model_params_disney
            )

    
    all_losses.append(loss.item())
    if len(all_abs_diff_per_params) == 0:
        all_abs_diff_per_params = np.abs(pred_params.cpu().detach().numpy() - model_params_disney.cpu().detach().numpy())
    else:
        for dists in np.abs(pred_params.cpu().detach().numpy() - model_params_disney.cpu().detach().numpy()):
            all_abs_diff_per_params = np.vstack((all_abs_diff_per_params, dists))

print(f"Average loss over {N_tries} tries: {np.mean(all_losses):.6f}")

print(f"Average absolute difference per parameter over {N_tries} tries: {np.mean(all_abs_diff_per_params, axis=0)}")

print(f"Standard deviation of absolute difference per Disney parameter over {N_tries} tries: {np.std(all_abs_diff_per_params, axis=0)}")

print(f"Min and max absolute difference per Disney parameter over {N_tries} tries: {np.min(all_abs_diff_per_params, axis=0)}, {np.max(all_abs_diff_per_params, axis=0)}")