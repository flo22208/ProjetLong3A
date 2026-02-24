from functools import partial
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
from merlDB.database import LABELS_MAT

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
    def __init__(self, img_size=(90, 90, 180), patch_size=(15, 15, 15), in_chans=3, embed_dim=768):
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
# MAE Class
# --------------------------------------------------------------------------
class MaskedAutoencoderViT3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone for 3D Data
    """
    def __init__(self, img_size=(90, 90, 180), patch_size=(15, 15, 15), in_chans=3,
                 embed_dim=12, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, poids=1, norme=2, smoothness_weight=0.0):
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

        self.nb_classes = len(LABELS_MAT)
        ## tableau de nombre de classes loss
        cluster_init = np.zeros((self.nb_classes, embed_dim))
        # must have nb_classes <= embed_dim
        for i in range(self.nb_classes) : # each class has a cluster center initialized to 1 in the corresponding dimension and 0 elsewhere
            cluster_init[i][i] = 1 # for equidistance initialization
        # Convertir en Parameter PyTorch pour qu'il soit sur le bon device
        self.cluster = nn.Parameter(torch.from_numpy(cluster_init).float(), requires_grad=False)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        def get_ponderation(size) :
            f = lambda a, b, i, j, k: 1 if (i < 30 or j < 30 or k < 30 or abs(i - size[2]) < 30 or abs(j - size[3]) < 30 or abs(k - size[4]) < 30) else poids
            ponderation = torch.Tensor(np.fromfunction(np.vectorize(f), size, dtype=float))
            return ponderation.to(device='cuda')
        self.ponderation = get_ponderation((1,3,90,90,180))

        self.sigmoid = nn.Sigmoid()
        self.norme = norme
        self.smoothness_weight = smoothness_weight
        
        # Créer la grille d'angles pour la transformation BRDF-to-RGB
        # img_size = (theta_d, theta_h, phi_d) = (90, 90, 180)
        theta_d_range = torch.linspace(0, np.pi/2, img_size[0])  # [0, pi/2] avec 90 valeurs
        theta_h_range = torch.linspace(0, np.pi/2, img_size[1])  # [0, pi/2] avec 90 valeurs
        phi_d_range = torch.linspace(0, np.pi, img_size[2])      # [0, pi] avec 180 valeurs
        
        # Créer une grille meshgrid: (theta_d, theta_h, phi_d)
        theta_d_grid, theta_h_grid, phi_d_grid = torch.meshgrid(theta_d_range, theta_h_range, phi_d_range, indexing='ij')
        
        # Calculer wiz = cos(theta_d)*cos(theta_h) - sin(theta_d)*cos(phi_d)*sin(theta_h)
        # Ce terme représente le cosinus de l'angle entre la normale et la lumière incidente
        wiz = (torch.cos(theta_d_grid) * torch.cos(theta_h_grid) - 
               torch.sin(theta_d_grid) * torch.cos(phi_d_grid) * torch.sin(theta_h_grid))
        
        # Clipper wiz entre [0, 1] et ajouter dimension batch et channels: (1, 1, D, H, W)
        self.wiz_grid = torch.clamp(wiz, 0, 1).unsqueeze(0).unsqueeze(0).to(device='cuda')

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, D, H, W)
        x: (N, L, patch_vol * 3)
        """
        p_d, p_h, p_w = self.patch_embed.patch_size
        
        assert imgs.shape[2] % p_d == 0 and imgs.shape[3] % p_h == 0 and imgs.shape[4] % p_w == 0

        n_d = imgs.shape[2] // p_d
        n_h = imgs.shape[3] // p_h
        n_w = imgs.shape[4] // p_w

        # Reshape to (N, 3, Grid_D, Patch_D, Grid_H, Patch_H, Grid_W, Patch_W)
        x = imgs.reshape(shape=(imgs.shape[0], 3, n_d, p_d, n_h, p_h, n_w, p_w))
        
        # Permute to (N, Grid_D, Grid_H, Grid_W, Patch_D, Patch_H, Patch_W, 3)
        x = torch.einsum('ncdthpwq->ndhptwqc', x)
        
        # Flatten to (N, Num_Patches, Patch_Vol * 3)
        x = x.reshape(shape=(imgs.shape[0], n_d * n_h * n_w, p_d * p_h * p_w * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_vol * 3)
        imgs: (N, 3, D, H, W)
        """
        p_d, p_h, p_w = self.patch_embed.patch_size
        n_d, n_h, n_w = self.patch_embed.grid_size
        
        assert x.shape[1] == n_d * n_h * n_w

        # Reshape to (N, Grid_D, Grid_H, Grid_W, Patch_D, Patch_H, Patch_W, 3)
        x = x.reshape(shape=(x.shape[0], n_d, n_h, n_w, p_d, p_h, p_w, 3))
        
        # Permute back to (N, 3, Grid_D, Patch_D, Grid_H, Patch_H, Grid_W, Patch_W)
        x = torch.einsum('ndhptwqc->ncdthpwq', x)
        
        # Reshape to final image
        imgs = x.reshape(shape=(x.shape[0], 3, n_d * p_d, n_h * p_h, n_w * p_w))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def supervised_loss(self, latent, labels, var=torch.sqrt(torch.tensor(1/2))):
        """
        Loss supervisée qui pousse le CLS token vers les centroïdes de classe
        
        Args:
            latent: (N, num_patches+1, D) - représentations latentes avec CLS token
            labels: (N,) - labels des classes
            var: variance pour la loss gaussienne
        
        Returns:
            loss: scalar - loss supervisée moyenne
        """
        # Extraire le CLS token (première position) qui représente l'échantillon entier
        cls_token = latent[:, 0, :]  # (N, D)
        
        # Récupérer les centroïdes correspondant aux labels
        mu_i = self.cluster[labels]  # (N, D)

        loss = ((cls_token - mu_i)**2) / (2 * var) 
    
        return loss.mean()


    def brdf_to_rgb(self, brdf_imgs):
        """
        Transforme les valeurs BRDF en RGB en appliquant le terme géométrique wiz.
        brdf_imgs: [N, 3, D, H, W] - valeurs BRDF
        returns: [N, 3, D, H, W] - valeurs RGB = BRDF * wiz
        """
        # self.wiz_grid est de taille (1, 1, D, H, W)
        # On le multiplie avec chaque channel RGB
        rgb = brdf_imgs * self.wiz_grid  # Broadcasting: (N, 3, D, H, W) * (1, 1, D, H, W)
        return rgb


    def smoothness_loss(self, imgs):
        """
        Calcule la loss de continuité/smoothness en pénalisant les gradients élevés.
        Calcule les différences entre pixels adjacents sur les 3 axes spatiaux.
        
        Args:
            imgs: [N, 3, D, H, W] - images (BRDF ou RGB)
        
        Returns:
            loss: scalar - moyenne des gradients sur les 3 axes
        """
        # Gradient le long de l'axe D (theta_d) - dimension 2
        # img.shape = [N, 3, 90, 90, 180]
        g1 = torch.sqrt(torch.mean((imgs[:, :, 1:, :, :] - imgs[:, :, :-1, :, :]) ** 2))

        # Gradient axe 2 (90)
        g2 = torch.sqrt(torch.mean((imgs[:, :, :, 1:, :] - imgs[:, :, :, :-1, :]) ** 2))

        # Gradient axe 3 (90)
        g3 = torch.sqrt(torch.mean((imgs[:, :, :, :, 1:] - imgs[:, :, :, :, :-1]) ** 2))
        
        # Moyenne pondérée (g1 a un poids double comme dans votre exemple)
        mean_grad = (g1 + g2 + 2*g3) / 4
        
        return mean_grad


    def forward_loss(self, imgs, pred, latent, mask,issupervised=False, labels=None):
        """
        imgs: [N, 3, D, H, W] - BRDF true
        pred: [N, L, patch_vol*3] - BRDF predicted
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)  # [N, L, patch_vol*3]
        loss_supervised = []
        
        # Variables pour stocker les images reconstruites (pour éviter duplication)
        pred_imgs = None
        pred_rgb = None
        
        if self.norme == 0:
            # Mean Absolute Logarithmic Error (MALE) avec transformation BRDF-to-RGB
            # 1. Reconstruire les images complètes depuis les patches
            target_imgs = self.unpatchify(target)  # [N, 3, D, H, W]
            pred_imgs = self.unpatchify(pred)      # [N, 3, D, H, W]
            
            # 2. Transformer BRDF en RGB avec le terme géométrique
            target_rgb = self.brdf_to_rgb(target_imgs)  # [N, 3, D, H, W]
            pred_rgb = self.brdf_to_rgb(pred_imgs)      # [N, 3, D, H, W]
            
            # 3. Calculer la loss logarithmique: |log(1 + rgb_true) - log(1 + rgb_pred)|
            loss_imgs = torch.abs(torch.log(torch.clamp(1 + target_rgb, min=1)) - torch.log(torch.clamp(1 + pred_rgb, min=1)))
            
            # 4. Appliquer la pondération
            loss_imgs = loss_imgs * self.ponderation
            
            # 5. Repatchifier pour appliquer le masque
            loss = self.patchify(loss_imgs)  # [N, L, patch_vol*3]
            
        else:
            # Loss standard (L1 ou L2) sans transformation RGB
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            if self.norme == 2:
                loss = (pred - target) ** 2
            elif self.norme == 1:
                loss = torch.abs(pred - target)
            else:
                assert False, f"norme={self.norme} non supportée. Utiliser 0 (log+RGB), 1 (L1) ou 2 (L2)"
            
            # Appliquer pondération
            loss_unpatch = self.unpatchify(loss)
            loss_unpatch = loss_unpatch * self.ponderation
            loss = self.patchify(loss_unpatch)

        if issupervised :
            loss_supervised.append(self.supervised_loss(latent, labels))
        
        # Calculer la loss moyenne par patch puis appliquer le masque
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()
        
        # Ajouter la loss de smoothness si activée
        if self.smoothness_weight > 0:
            # Reconstruire l'image prédite si pas déjà fait
            if pred_imgs is None:
                pred_imgs = self.unpatchify(pred)  # [N, 3, D, H, W]
            
            # Appliquer directement sur les valeurs BRDF
            smooth_loss = self.smoothness_loss(pred_imgs)
            
            # Ajouter la loss de smoothness pondérée
            loss = loss + self.smoothness_weight * smooth_loss
        
        return (loss, torch.stack(loss_supervised).mean()) if issupervised else loss
    
    def forward_clustering(self,imgs):
        """
        Permet de récupérer la classe d'un élément prédit à partir de sa représentation latente et des centroïdes de classe
        """
        latent = self.forward_features(imgs)
        cls_token = latent[:, 0, :]  
        # Calculer les distances entre le CLS token et les centroïdes de classe
        distances = torch.cdist(cls_token, self.cluster)  # (N, nb_classes)
        # Prédire la classe la plus proche
        predicted_classes = torch.argmin(distances, dim=1)  # (N,)
        return predicted_classes
    


    def forward(self, imgs, mask_ratio=0.75,ispermute=True,issupervised=False,labels=None):
        permute = torch.randperm(3)
        if ispermute :
            imgs = imgs[:,permute,:,:]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) 
        if issupervised :
            loss, loss_supervised = self.forward_loss(imgs, pred,latent, mask, issupervised=issupervised, labels=labels)
            return loss, loss_supervised, pred, mask, latent
        else :
            loss = self.forward_loss(imgs, pred,latent, mask, issupervised=issupervised, labels=labels)
            return loss, pred, mask, latent  
        
    
    
    def forward_features(self, x):
        """
        Permet de récupérer les représentations latentes
        """
        # 1. Embed patches
        x = self.patch_embed(x)

        # 2. Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]

        # 3. Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 4. Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


def mae_model_brdf(**kwargs):
    model = MaskedAutoencoderViT3D(
        img_size=(90, 90, 180), patch_size=(15, 15, 15), embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set arch
mae_brdf = mae_model_brdf  # decoder: 512 dim, 8 blocks