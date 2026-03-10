

import torch
import torch.nn as nn
import numpy as np

from model import EncoderViT3D
from numpyBRDF import BRDF, rusinkiewicz_to_LV

folder_brdfs = "brdfs_disney/"

epochs = 3000
embed_dim = 96
latent_space_dim = 8
where_zeros = [4, 7, 9, 11]
# where_zeros = [7, 11]
# where_zeros = []
batch_size = 20

## Create table of all angles

RES_THETA_H = 90
RES_THETA_D = 90
RES_PHI_D = 180

MAX_THETA_H = 90
MAX_THETA_D = 90
MAX_PHI_D = 180

theta_hs = np.deg2rad(np.linspace(0, RES_THETA_H, MAX_THETA_H))
theta_ds = np.deg2rad(np.linspace(0, RES_THETA_D, MAX_THETA_D))
phi_ds = np.deg2rad(np.linspace(0, RES_PHI_D, MAX_PHI_D))

model_file = f"results/encoder_disney_{embed_dim}_{latent_space_dim}_{epochs}_{batch_size}_0.0088.pt"

model = EncoderViT3D(embed_dim=embed_dim, latent_space_dim=latent_space_dim)        # instantiate architecture first
model.load_state_dict(torch.load(model_file))
model.to('cuda')
model.eval()

if __name__ == "__main__":
    #### Ground Truth : pick random Disney parameters, compute BRDF, save

    params_disney = torch.rand(1, 12, device='cuda')
    for i in where_zeros:
            params_disney[:, i:i+1] = 0.0
    model_params_disney = torch.zeros(1, latent_space_dim, device='cuda')
    cpt = 0
    for i in range(12):
        if not(params_disney[: , i:i+1].sum() == 0):
            model_params_disney[:, cpt:cpt+1] = params_disney[:, i:i+1]
            cpt +=1

    material_params = {
            "baseColor": params_disney[0, 0:3].cpu().numpy(),
            "metallic": params_disney[0, 3].item(),
            "subsurface": params_disney[0, 4].item(),
            "specular": params_disney[0, 5].item(),
            "roughness": params_disney[0, 6].item(),
            "specularTint": params_disney[0, 7].item(),
            "anisotropic": 0.0,
            "sheen": params_disney[0, 8].item(),
            "sheenTint": params_disney[0, 9].item(),
            "clearcoat": params_disney[0, 10].item(),
            "clearcoatGloss": params_disney[0, 11].item(),
        }
    print(f"gt params: {material_params}")
    
    brdf = np.zeros((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3), dtype=np.float32)

    for hi in range(MAX_THETA_H):
        for di in range(MAX_THETA_D):
            for pi in range(MAX_PHI_D):
                theta_h = theta_hs[hi]
                theta_d = theta_ds[di]
                phi_d = phi_ds[pi]

                L, V, N_vec, X, Y = rusinkiewicz_to_LV(
                    theta_h, 0, theta_d, phi_d
                )
                vals = BRDF(
                    L, V, N_vec, X, Y,
                    baseColor=material_params["baseColor"], metallic=material_params["metallic"], subsurface=material_params["subsurface"],
                    specular=material_params["specular"], roughness=material_params["roughness"], specularTint=material_params["specularTint"],
                    anisotropic=material_params["anisotropic"], sheen=material_params["sheen"], sheenTint=material_params["sheenTint"],
                    clearcoat=material_params["clearcoat"], clearcoatGloss=material_params["clearcoatGloss"]
                )
                if vals[0] != -1:
                    brdf[hi, di, pi] = vals

    brdf = brdf / (1 + brdf)

    np.savez(f"{folder_brdfs}/brdf_0.npz",
            params=material_params,
            brdf=brdf)

    #### Intermediary : convert the GT BRDF to torch
    rgbs_model = torch.tensor(brdf, device='cuda', dtype=torch.float32).unsqueeze(0)  # add batch dimension
    rgbs_model = torch.einsum('bdhwc->bcdhw', rgbs_model)

    #### Prediction : pass through model, compute BRDF, save
    with torch.no_grad():
        output = model(
                    rgbs_model, model_params_disney
                )

    params = output[1].cpu().numpy()[0]
    material_params = {
            "baseColor": params[0:3],
            "metallic": params[3],
            "subsurface": 0.0,
            "specular": params[4],
            "roughness": params[5],
            "specularTint": 0.0,
            "anisotropic": 0.0,
            "sheen": params[6],
            "sheenTint": 0.0,
            "clearcoat": params[7],
            "clearcoatGloss": 0.0,
        }
    print(f"pred params: {material_params}")

    brdf = np.zeros((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3), dtype=np.float32)

    for hi in range(MAX_THETA_H):
        for di in range(MAX_THETA_D):
            for pi in range(MAX_PHI_D):
                theta_h = theta_hs[hi]
                theta_d = theta_ds[di]
                phi_d = phi_ds[pi]

                L, V, N_vec, X, Y = rusinkiewicz_to_LV(
                    theta_h, 0, theta_d, phi_d
                )
                vals = BRDF(
                    L, V, N_vec, X, Y,
                    baseColor=material_params["baseColor"], metallic=material_params["metallic"], subsurface=material_params["subsurface"],
                    specular=material_params["specular"], roughness=material_params["roughness"], specularTint=material_params["specularTint"],
                    anisotropic=material_params["anisotropic"], sheen=material_params["sheen"], sheenTint=material_params["sheenTint"],
                    clearcoat=material_params["clearcoat"], clearcoatGloss=material_params["clearcoatGloss"]
                )
                if vals[0] != -1:
                    brdf[hi, di, pi] = vals

    # Tonemapping
    brdf = brdf / (1 + brdf)
                
    np.savez(f"{folder_brdfs}/brdf_1.npz",
            params=material_params,
            brdf=brdf)
    
    #### 

    print("Done")