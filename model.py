from functools import partial
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
from torchBRDF import BRDF, rusinkiewicz_to_LV

# --------------------------------------------------------------------------
# 3D Position Embedding Helper
# --------------------------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple of (grid_d, grid_h, grid_w)
    out: (grid_d*grid_h*grid_w, embed_dim) or with cls_token (grid_d*grid_h*grid_w+1, embed_dim)
    """
    grid_d, grid_h, grid_w = grid_size
    
    d_embed = embed_dim // 3
    h_embed = embed_dim // 3
    w_embed = embed_dim - d_embed - h_embed 

    grid_k = np.arange(grid_d, dtype=np.float32)
    grid_i = np.arange(grid_h, dtype=np.float32)
    grid_j = np.arange(grid_w, dtype=np.float32)

    grid = np.meshgrid(grid_k, grid_i, grid_j, indexing='ij') 
    
    grid_k = grid[0].reshape(-1)
    grid_i = grid[1].reshape(-1)
    grid_j = grid[2].reshape(-1)

    pos_embed_k = get_1d_sincos_pos_embed_from_grid(d_embed, grid_k)
    pos_embed_i = get_1d_sincos_pos_embed_from_grid(h_embed, grid_i)
    pos_embed_j = get_1d_sincos_pos_embed_from_grid(w_embed, grid_j)

    pos_embed = np.concatenate([pos_embed_k, pos_embed_i, pos_embed_j], axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

# --------------------------------------------------------------------------
# 3D Patch Embedding Layer
# --------------------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    """ 3D Volume to Patch Embedding
    """
    def __init__(self, img_size=(90, 90, 180), patch_size=(90, 90, 180), in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Use Conv3d to project patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, Embed, Grid_D, Grid_H, Grid_W)
        x = x.flatten(2).transpose(1, 2)  # (B, Num_Patches, Embed)
        return x

# --------------------------------------------------------------------------
# Encoder Class
# --------------------------------------------------------------------------
class EncoderViT3D(nn.Module):
    """ Encoder with VisionTransformer backbone for 3D Data
    """
    def __init__(self, img_size=(90, 90, 180), patch_size=(15, 15, 15), in_chans=3,
                 embed_dim=12, depth=12, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.loss = nn.MSELoss()
        self.initialize_weights()


    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, gt_params, pred_params):
        """
        gt_params: [N, 12]
        pred_params: [N, 12]
        """
        loss = self.loss(gt_params, pred_params)
        return loss


    def forward(self, brdfs, gt_params):
        pred_params = self.forward_encoder(brdfs)
        pred_params_cls_token = pred_params[:, :1, :].squeeze(1)
        pred_params_cls_token = torch.sigmoid(pred_params_cls_token)
        loss = self.forward_loss(gt_params, pred_params_cls_token)
        return loss, pred_params_cls_token

if __name__ == '__main__':
    wi, wo, N = rusinkiewicz_to_LV('cuda')

    params = torch.tensor(
        [[
            0.82, 0.67, 0.16,   # baseColor (R,G,B)
            0.0,                # metallic
            0.0,                # subsurface
            0.5,                # specular
            0.5,                # roughness
            0.0,                # specularTint
            0.0,                # sheen
            0.5,                # sheenTint
            0.0,                # clearcoat
            1.0                 # clearcoatGloss
        ]],
        dtype=torch.float32,
        device='cuda'
    )

    torch_output = BRDF(params, wi, wo, N)
    mask = torch.isinf(torch_output)
    torch_output = torch_output / (1 + torch_output)
    torch_output[mask] = 1.0
    # torch_output = 1 - torch.exp(-torch_output)
    print(torch.isnan(torch_output).any())
    print(torch_output.min(), torch_output.max())
    torch_output = torch.einsum('bdhwc->bcdhw', torch_output)

    # repeat 12 times
    torch_output = torch.repeat_interleave(torch_output, 12, dim=0)

    print(torch_output.shape)

    params = torch.repeat_interleave(params, 12, dim=0)

    # set arch
    mae_brdf = EncoderViT3D() # decoder: 512 dim, 8 blocks
    mae_brdf.to('cuda')

    print(mae_brdf(torch_output, params))