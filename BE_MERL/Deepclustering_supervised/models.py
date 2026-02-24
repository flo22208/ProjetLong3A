
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np
from sklearn.cluster import KMeans

from mae_model_brdf import MaskedAutoencoderViT3D



class MERLClusteringLayer(nn.Module):
    """
    Couche de clustering pour MERL qui apprend des centroïdes de classes.
    """
    def __init__(self, n_classes=30, latent_dim=768, alpha=1.0):
        super(MERLClusteringLayer, self).__init__()
        
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.alpha = alpha
        
        # Centroïdes des classes
        self.centroids = nn.Parameter(torch.Tensor(n_classes, latent_dim))
        nn.init.xavier_uniform_(self.centroids)
        
    def forward(self, z):
        """Calcule la soft assignment pour chaque classe"""
        # z: (batch_size, latent_dim)
        z_expanded = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        centroids_expanded = self.centroids.unsqueeze(0)  # (1, n_classes, latent_dim)
        
        # Distance euclidienne
        distances = torch.sum((z_expanded - centroids_expanded) ** 2, dim=2)
        
        # Distribution Student's t
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q
    



class MAEWithClustering(nn.Module):
    """
    Modèle MAE + Clustering pour MERL
    """
    def __init__(self, mae, n_classes=4, latent_dim=768):
        super(MAEWithClustering, self).__init__()
        self.mae = mae
        self.n_classes = n_classes
        self.expected_latent_dim = latent_dim
        self.clustering_layer = MERLClusteringLayer(n_classes=n_classes, latent_dim=latent_dim)
    
    def initialize_centroids_kmeans(self, data_loader, device, n_samples=1000):
        """
        Initialise les centroïdes avec k-means sur les représentations latentes
        
        Args:
            data_loader: DataLoader avec les données d'entraînement
            device: Device (cuda ou cpu)
            n_samples: Nombre maximum d'échantillons à utiliser
        """
        self.eval()
        
        all_latents = []
        with torch.no_grad():
            for batch_data in data_loader:
                X = batch_data['values'].to(device)
                
                mae_output = self.mae(X)
                if isinstance(mae_output, tuple) and len(mae_output) >= 4:
                    latent = mae_output[3]
                else:
                    latent = mae_output
                
                # Réduire à 2D si nécessaire
                if latent.dim() > 2:
                    latent = torch.mean(latent, dim=1)
                
                all_latents.append(latent.cpu().numpy())
                
                if len(all_latents) * X.shape[0] >= n_samples:
                    break
        
        latents = np.concatenate(all_latents, axis=0)[:n_samples]
        
        # K-means
        print(f"\nInitializing centroids with K-means on {latents.shape[0]} samples...")
        print(f"Latent space dimension: {latents.shape[1]}")
        kmeans = KMeans(n_clusters=self.n_classes, n_init=10, random_state=42)
        kmeans.fit(latents)
        
        # Mettre à jour les centroïdes
        centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
        self.clustering_layer.centroids.data = centroids
        
        print(f"✓ Centroids initialized with K-means")
        
    def forward(self, x):
        """Forward pass MAE + Clustering"""
        # MAE forward (retourne: loss, output, mask, latent)
        mae_output = self.mae(x)
        
        if isinstance(mae_output, tuple) and len(mae_output) >= 4:
            loss, out, mask, latent = mae_output[0], mae_output[1], mae_output[2], mae_output[3]
        else:
            # Si MAE retourne un tenseur simple, l'utiliser comme latent
            latent = mae_output
            loss = None
            out = None
            mask = None
        
        # Réduire latent à 2D en utilisant average pooling au lieu de flatten
        if latent.dim() > 2:
            # Si latent a plus de 2 dimensions (batch_size, seq_len, embed_dim)
            # Utiliser average pooling sur la dimension séquence
            latent = torch.mean(latent, dim=1)  # (batch_size, embed_dim)
        
        # Clustering sur les représentations latentes
        q = self.clustering_layer(latent)
        
        return loss, out, mask, latent, q
    
