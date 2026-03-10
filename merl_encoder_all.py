import os
from model import EncoderViT3D
import torch
import random
from merlDB import database as db
import numpy as np

output_folder = "results/merl_on_encoder/"
os.makedirs(output_folder, exist_ok=True)

epochs = 6000
embed_dim = 96
latent_space_dim = 8
where_zeros = [4, 7, 9, 11]
# where_zeros = [7, 11]
# where_zeros = []
batch_size = 20

model_file = f"results/encoder_disney_{embed_dim}_{latent_space_dim}_{epochs}_{batch_size}_0.0064.pt"

## load model for .pt file
model = EncoderViT3D(embed_dim=embed_dim, latent_space_dim=latent_space_dim)        # instantiate architecture first
model.load_state_dict(torch.load(model_file))
model.to('cuda')
model.eval()


## load BRDF from MERL database
dbuilder = db.DBuilder(interp_method="linear",db_path='merlDB/db/brdfs/')
ldb = dbuilder.list_db()

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

angles = np.zeros((MAX_THETA_H*MAX_THETA_D*MAX_PHI_D, 3))

# fill angles with all tripelts of theta_h, theta_d, phi_d
idx = 0
for hi in range(MAX_THETA_H):
    for di in range(MAX_THETA_D):
        for pi in range(MAX_PHI_D):
            angles[idx] = [phi_ds[pi], theta_ds[di], theta_hs[hi]]
            idx += 1

all_material_params = np.array([])

for i in range(len(ldb)):
    mat = ldb[i]
    dbuilder.load_mat(mat)
    print("Loaded " + mat)
    brdf = dbuilder.brdf_function(mat)
    name = mat

    rgbs = brdf(angles)
    rgbs = rgbs / (1.0 + rgbs)  # tonemapping

    ## Resize to MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3
    rgbs = rgbs.reshape((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3))

    # print(rgbs)

    ## Convert to torch tensor
    rgbs = torch.tensor(rgbs, dtype=torch.float32)
    rgbs = rgbs.permute(3, 0, 1, 2)   # C D H W
    rgbs = rgbs.unsqueeze(0)         # 1 C D H W

    ## Pass through model
    params = torch.rand(1, latent_space_dim,  device='cuda', dtype=torch.float32)
    with torch.no_grad():
        output = model(rgbs.to('cuda'), params)

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

    entry = (name, material_params)

    all_material_params = np.append(all_material_params, entry)

np.savez(f"{output_folder}/{embed_dim}_{latent_space_dim}_{epochs}_{batch_size}.npz", params=all_material_params)

# read file and display one material
data = np.load(f"{output_folder}/{embed_dim}_{latent_space_dim}_{epochs}_{batch_size}.npz", allow_pickle=True)
print(data['params'][0])
print(data['params'][1])