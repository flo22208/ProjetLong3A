from model import EncoderViT3D
import torch
import random
from merlDB import database as db
import numpy as np

epochs = 3000
embed_dim = 96
latent_space_dim = 10
where_zeros = [4, 7, 9, 11]

## load model for .pt file
model = EncoderViT3D(embed_dim=embed_dim, latent_space_dim=latent_space_dim)        # instantiate architecture first
model.load_state_dict(torch.load("results/encoder_disney_96_10_3000_20_0.0096.pt"))
model.to('cuda')
model.eval()


## load BRDF from MERL database
dbuilder = db.DBuilder(interp_method="linear",db_path='merlDB/db/brdfs/')
ldb = dbuilder.list_db()
mat = ldb[1]
mat = "blue-acrylic"
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

rgbs = brdf(angles)
rgbs = 1.0 / (1.0 + rgbs)  # tonemapping

## Resize to MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3
rgbs = rgbs.reshape((MAX_THETA_H, MAX_THETA_D, MAX_PHI_D, 3))

print(rgbs)

## Convert to torch tensor
rgbs = torch.tensor(rgbs, dtype=torch.float32)
rgbs = rgbs.permute(3, 0, 1, 2)   # C D H W
rgbs = rgbs.unsqueeze(0)         # 1 C D H W 
## Pass through model
params = torch.tensor(
        [[
            0.82, 0.67, 0.16,   # baseColor (R,G,B)
            0.0,                # metallic
            0.0,                # subsurface
            0.5,                # specular
            0.5,                # roughness
#            0.0,                # specularTint
            0.0,                # sheen
            0.5,                # sheenTint
            0.0,                # clearcoat
#            1.0                 # clearcoatGloss
        ]],
        dtype=torch.float32,
        device='cuda'
    )
with torch.no_grad():
    output = model(rgbs.to('cuda'), params)

print(output)