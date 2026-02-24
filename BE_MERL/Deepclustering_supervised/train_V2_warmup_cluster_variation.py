import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from data_loader import load_merl_datasets, save_dataset_info
from neural_data import MAETrainer

from visualizer import visualize_latent_space_pca
from models import MAEWithClustering

try:
    from mae_model_brdf import MaskedAutoencoderViT3D
except ImportError:
    print("Warning: mae_model_brdf not found")
    MaskedAutoencoderViT3D = None






def setup_output_dirs(outdir, checkpoint_dir):
    """Crée les répertoires de sortie"""
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(model, optimizer, checkpoint_dir, epoch, best_acc):
    """Sauvegarde un checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Sauvegarder aussi le dernier
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)



def target_distribution(q):
    """Calcule la distribution cible pour le clustering"""
    # q shape: (batch_size, n_classes)
    weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-10)  # Éviter division par zéro
    p = weight / (weight.sum(dim=1, keepdim=True) + 1e-10)  # Normaliser par sample
    return p



def combined_merl_loss(mae_loss, q,
                       mae_weight=1.0, cluster_weight=0.5, supervised_weight=1.0):
    """
    Loss combinée: MAE + Clustering
    """
    # Loss MAE (déjà fournie par le modèle)
    mae_loss_val = mae_loss if mae_loss is not None else torch.tensor(0.0)
    
    # Loss clustering (KL divergence avec distribution cible)
    p = target_distribution(q.detach())
    
    # KL divergence correcte sur probabilités avec epsilon pour stabilité
    eps = 1e-8
    cluster_loss = nn.functional.kl_div(
        torch.log(q + eps), 
        p, 
        reduction='batchmean'
    )
    
    
    # Loss totale
    total_loss = (mae_weight * mae_loss_val + 
                  cluster_weight * cluster_loss )
    
    return total_loss, mae_loss_val, cluster_loss




def train_epoch_merl(model, train_loader, optimizer, device, mae_loss_weight, cluster_loss_weight):
    """Entraîne le modèle MAE+Clustering pour une epoch"""
    model.train()
    
    total_loss = 0
    total_mae_loss = 0
    total_cluster_loss = 0
    
    all_preds = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_data in pbar:
        X = batch_data['values'].to(device)
        
        # Forward pass
        mae_loss, out, mask, latent, q = model(X)
        
        # Calcul de la loss combinée
        loss, mae_l, cluster_l = combined_merl_loss(
            mae_loss, q,
            mae_weight=mae_loss_weight,
            cluster_weight=cluster_loss_weight
        )
        
        # Prédictions
        preds = torch.argmax(q, dim=1).cpu().numpy()
        all_preds.extend(preds)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        total_loss += loss.item()
        total_mae_loss += mae_l.item() if isinstance(mae_l, torch.Tensor) else mae_l
        total_cluster_loss += cluster_l.item()
        
  
        
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_mae_loss = total_mae_loss / len(train_loader)
    avg_cluster_loss = total_cluster_loss / len(train_loader)
    
    n_unique_preds = len(np.unique(all_preds))
    
    return avg_loss, avg_mae_loss, avg_cluster_loss, n_unique_preds


def get_final_predictions(model, train_loader, test_loader, device):
    """
    Récupère les prédictions de clustering finales pour tous les samples
    """
    model.eval()
    
    train_preds = []
    train_latents = []
    
    with torch.no_grad():
        for batch_data in train_loader:
            X = batch_data['values'].to(device)
            _, _, _, latent, q = model(X)
            
            preds = torch.argmax(q, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_latents.append(latent.cpu().numpy())
    
    train_preds = np.array(train_preds)
    train_latents = np.concatenate(train_latents, axis=0)
    
    test_preds = None
    test_latents = None
    
    if test_loader is not None:
        test_preds = []
        test_latents = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                X = batch_data['values'].to(device)
                _, _, _, latent, q = model(X)
                
                preds = torch.argmax(q, dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_latents.append(latent.cpu().numpy())
        
        test_preds = np.array(test_preds)
        test_latents = np.concatenate(test_latents, axis=0)
    
    return train_preds, train_latents, test_preds, test_latents


def merge_close_clusters(model, train_loader, device, merge_threshold=0.5):
    """
    Fusionne les clusters dont les centroïdes sont à une distance inférieure au seuil.
    
    Args:
        model: Le modèle avec clustering
        train_loader: DataLoader pour calculer les statistiques
        device: Device (cuda/cpu)
        merge_threshold: Seuil de distance pour fusionner (distance normalisée)
    
    Returns:
        Number of merges performed
    """
    model.eval()
    
    # Récupérer les centroïdes actuels
    centroids = model.clustering_layer.centroids.data.clone()
    n_clusters = centroids.shape[0]
    
    # Calculer la matrice des distances entre centroïdes
    distances = torch.cdist(centroids, centroids, p=2)
    
    # Normaliser les distances par la distance moyenne
    mean_dist = distances[distances > 0].mean()
    normalized_distances = distances / (mean_dist + 1e-8)
    
    # Identifier les paires de clusters à fusionner
    merge_pairs = []
    merged_clusters = set()
    
    for i in range(n_clusters):
        if i in merged_clusters:
            continue
        for j in range(i + 1, n_clusters):
            if j in merged_clusters:
                continue
            if normalized_distances[i, j] < merge_threshold:
                merge_pairs.append((i, j))
                merged_clusters.add(j)
    
    if len(merge_pairs) == 0:
        return 0
    
    # Créer un mapping des anciens clusters vers les nouveaux
    cluster_mapping = {i: i for i in range(n_clusters)}
    for i, j in merge_pairs:
        cluster_mapping[j] = cluster_mapping[i]
    
    # Recalculer les nouveaux centroïdes en moyennant les clusters fusionnés
    new_centroids = []
    processed = set()
    
    for i in range(n_clusters):
        target = cluster_mapping[i]
        if target in processed:
            continue
        
        # Trouver tous les clusters qui mappent vers target
        group = [k for k in range(n_clusters) if cluster_mapping[k] == target]
        
        # Moyenne des centroïdes du groupe
        group_centroids = centroids[group]
        new_centroid = group_centroids.mean(dim=0)
        new_centroids.append(new_centroid)
        processed.add(target)
    
    new_centroids = torch.stack(new_centroids)
    
    # Mettre à jour les centroïdes du modèle
    old_n_clusters = model.clustering_layer.n_classes
    new_n_clusters = new_centroids.shape[0]
    
    # Créer une nouvelle couche de clustering avec le bon nombre de clusters
    from models import MERLClusteringLayer
    new_clustering_layer = MERLClusteringLayer(
        n_classes=new_n_clusters,
        latent_dim=model.clustering_layer.latent_dim,
        alpha=model.clustering_layer.alpha
    ).to(device)
    
    # Copier les nouveaux centroïdes
    new_clustering_layer.centroids.data = new_centroids
    model.clustering_layer = new_clustering_layer
    
    print(f"  → Merged {old_n_clusters} → {new_n_clusters} clusters ({len(merge_pairs)} merges)")
    
    return len(merge_pairs)


def split_dispersed_clusters(model, train_loader, device, split_threshold=2.0, n_samples=1000):
    """
    Divise les clusters avec une variance interne élevée.
    
    Args:
        model: Le modèle avec clustering
        train_loader: DataLoader pour calculer les statistiques
        device: Device (cuda/cpu)
        split_threshold: Seuil de variance normalisée pour diviser
        n_samples: Nombre d'échantillons à utiliser pour l'analyse
    
    Returns:
        Number of splits performed
    """
    model.eval()
    
    # Collecter les latents et leurs assignations
    latents_list = []
    assignments_list = []
    
    with torch.no_grad():
        n_collected = 0
        for batch_data in train_loader:
            if n_collected >= n_samples:
                break
            
            X = batch_data['values'].to(device)
            _, _, _, latent, q = model(X)
            
            assignments = torch.argmax(q, dim=1)
            
            latents_list.append(latent)
            assignments_list.append(assignments)
            
            n_collected += latent.shape[0]
    
    latents = torch.cat(latents_list, dim=0)
    assignments = torch.cat(assignments_list, dim=0)
    
    n_clusters = model.clustering_layer.n_classes
    centroids = model.clustering_layer.centroids.data
    
    # Calculer la variance intra-cluster pour chaque cluster
    cluster_variances = []
    clusters_to_split = []
    
    for cluster_id in range(n_clusters):
        mask = assignments == cluster_id
        if mask.sum() < 10:  # Trop peu de samples
            cluster_variances.append(0.0)
            continue
        
        cluster_points = latents[mask]
        centroid = centroids[cluster_id]
        
        # Variance = distance moyenne au centroïde
        distances = torch.norm(cluster_points - centroid, dim=1)
        variance = distances.mean().item()
        cluster_variances.append(variance)
    
    # Normaliser les variances
    mean_variance = np.mean([v for v in cluster_variances if v > 0])
    normalized_variances = [v / (mean_variance + 1e-8) for v in cluster_variances]
    
    # Identifier les clusters à diviser
    for cluster_id, norm_var in enumerate(normalized_variances):
        if norm_var > split_threshold:
            mask = assignments == cluster_id
            if mask.sum() >= 20:  # Assez de points pour diviser
                clusters_to_split.append(cluster_id)
    
    if len(clusters_to_split) == 0:
        return 0
    
    # Diviser chaque cluster identifié en 2 avec K-means
    from sklearn.cluster import KMeans
    
    new_centroids_list = [centroids[i] for i in range(n_clusters) if i not in clusters_to_split]
    
    for cluster_id in clusters_to_split:
        mask = assignments == cluster_id
        cluster_points = latents[mask].cpu().numpy()
        
        # K-means avec 2 clusters
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        kmeans.fit(cluster_points)
        
        # Ajouter les deux nouveaux centroïdes
        for new_centroid in kmeans.cluster_centers_:
            new_centroids_list.append(torch.from_numpy(new_centroid).float().to(device))
    
    new_centroids = torch.stack(new_centroids_list)
    
    # Mettre à jour le modèle
    old_n_clusters = model.clustering_layer.n_classes
    new_n_clusters = new_centroids.shape[0]
    
    from models import MERLClusteringLayer
    new_clustering_layer = MERLClusteringLayer(
        n_classes=new_n_clusters,
        latent_dim=model.clustering_layer.latent_dim,
        alpha=model.clustering_layer.alpha
    ).to(device)
    
    new_clustering_layer.centroids.data = new_centroids
    model.clustering_layer = new_clustering_layer
    
    print(f"  → Split {old_n_clusters} → {new_n_clusters} clusters ({len(clusters_to_split)} splits)")
    
    return len(clusters_to_split)


def adapt_clusters(model, train_loader, device, merge_threshold=0.5, split_threshold=2.0, n_samples=1000):
    """
    Adapte le nombre de clusters en fusionnant les proches et divisant les dispersés.
    
    Args:
        model: Le modèle avec clustering
        train_loader: DataLoader pour l'analyse
        device: Device
        merge_threshold: Seuil de distance normalisée pour fusion (plus petit = fusion agressive)
        split_threshold: Seuil de variance normalisée pour division (plus petit = division agressive)
        n_samples: Nombre d'échantillons pour l'analyse
    
    Returns:
        Dict avec statistiques de l'adaptation
    """
    initial_n_clusters = model.clustering_layer.n_classes
    
    # 1. Fusionner les clusters proches
    n_merges = merge_close_clusters(model, train_loader, device, merge_threshold)
    
    # 2. Diviser les clusters dispersés
    n_splits = split_dispersed_clusters(model, train_loader, device, split_threshold, n_samples)
    
    final_n_clusters = model.clustering_layer.n_classes
    
    return {
        'initial_clusters': initial_n_clusters,
        'final_clusters': final_n_clusters,
        'n_merges': n_merges,
        'n_splits': n_splits
    }








if __name__ == "__main__":
      
    # Paramètres du modèle MAE
    dim_latent = 768#768                # Dimension de l'espace latent
    mask_ratio = 0.5                # Ratio de masquage
    poids = 2                        # Poids pour la perte MAE
    norme = 2                        # Norme de la fonction de perte (1 ou 2)
    
    # Paramètres d'entraînement
    epochs = 200                     # Nombre d'epochs
    batch_size = 8                  # Batch size AUGMENTÉ de 4 à 8
    lr_init_mae = 1e-4                       # Learning rate CORRIGÉ de 1e-1 à 1e-4
    lr_init_clustering = 0               # Learning rate pour le clustering
    lr_after_warmup_mae = 1e-5          # Learning rate après warmup pour MAE
    lr_after_warmup_clustering = 1e-3   # Learning rate après warmup pour clustering
    weight_decay = 0.            # Weight decay
    
    # Démarrer le clustering après N epochs
    warmup_epochs = 100
    
    # Paramètres d'adaptation des clusters
    adapt_clusters_every = 3           # Adapter les clusters tous les N epochs (après warmup)
    merge_threshold = 0.08               # Seuil de fusion: distance normalisée (0.3-0.7 typique)
    split_threshold = 2000000               # Seuil de division: variance normalisée (1.5-3.0 typique)
    
    # Poids des losses - RÉÉQUILIBRÉ
    mae_loss_weight_init = 1.0            # Poids de la loss MAE
    cluster_loss_weight_init = 0#0.3#0.5        # Poids de la loss clustering RÉDUIT de 2.0 à 1.0
    mae_loss_weight_after_warmup = 1.0     # Poids de la loss MAE après warmup
    cluster_loss_weight_after_warmup = 0.5  # Poids de la loss clustering après warmup
    
    # Chemins et répertoires
    outdir = 'results_warmup/'       # Répertoire de sortie (warmup approach)
    checkpoint_dir = 'checkpoints_warmup/'  # Répertoire pour les checkpoints

    
    
    # Données MERL
    merldir = 'merlDB/db/brdfs/'     # Chemin vers la base de données MERL
    mediandir = 'merlDB/db/'         # Chemin vers le fichier median MERL
    train_size = 100                 # Nombre de matériaux pour l'entraînement
    monomat = None                   # Si spécifié, charge uniquement ce matériau
    
    # Autres paramètres
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device (cuda ou cpu)
    seed = 0                         # Graine aléatoire

     # Configuration
    print("="*70)
    print("MAE + CLUSTERING TRAINING ON MERL")
    print("="*70)
    
    # Device
    device = torch.device(device_name)
    print(f"\nDevice: {device}")
    
    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Créer les répertoires
    setup_output_dirs(outdir, checkpoint_dir)
    
    # Charger les données MERL
    print("\n" + "="*70)
    print("LOADING MERL DATASETS")
    print("="*70)
    
    # Créer un objet args pour load_merl_datasets
    class Args:
        def __init__(self):
            self.outdir = outdir
            self.merldir = merldir
            self.mediandir = mediandir
            self.train_size = train_size
            self.monomat = monomat
            self.seed = seed
            self.load_previous = False
            self.materiau = ""
    
    args_data = Args()
    data = load_merl_datasets(args=args_data, device=device)
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_mats = data['train_mats']
    test_mats = data['test_mats']
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")
    
    # Nombre de classes = nombre de catégories définies dans CLASSES
    
    # Créer le modèle MAE
    print("\n" + "="*70)
    print("CREATING MAE MODEL")
    print("="*70)
    
    if MaskedAutoencoderViT3D is None:
        raise ImportError("mae_model_brdf not available")
    
    mae = MaskedAutoencoderViT3D(
        poids=poids,
        embed_dim=dim_latent,
        norme=norme
    )
    mae.to(device)
    
    # Wrapper avec clustering
    nb_clusters_init = 99  # Nombre initial de clusters (peut être ajusté)
    model = MAEWithClustering(mae, latent_dim=dim_latent, n_classes=nb_clusters_init)
    model.to(device)
    
    print(f"MAE Dimension: {dim_latent}")
    print(f"Mask Ratio: {mask_ratio}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimiseur et scheduler
    optimizer = optim.Adam([
        {'params': model.mae.parameters(), 'lr': lr_init_mae},
        {'params': model.clustering_layer.parameters(), 'lr': lr_init_clustering}
    ], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    
    
    train_loader = torchdata.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    test_loader = None
    if test_dataset:
        test_loader = torchdata.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    
    # (Ne pas init les centroïdes avant le warmup)
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(outdir, f'logs_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    # Entraînement
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_accuracy = 0
    losses_history = {
        'total': [],
        'mae': [],
        'cluster': [],
        'supervised': [],
        'train_acc': [],
        'test_acc': []
    }
    
    cluster_evolution = []  # Historique du nombre de clusters
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)

        # Init des centroïdes avec K-means au début du clustering
        if epoch == warmup_epochs:
            print(f"\n{'='*70}")
            print(f"Initializing centroids with K-means before starting clustering")
            print(f"{'='*70}")
            model.initialize_centroids_kmeans(train_loader, device, n_samples=1000)
            
            ## mettre le learning rate du cluster
            for param_group in optimizer.param_groups:
                if param_group['lr'] == 0:
                    param_group['lr'] = lr_after_warmup_clustering
                else:
                    param_group['lr'] = lr_after_warmup_mae
            mae_loss_weight_init = mae_loss_weight_after_warmup
            cluster_loss_weight_init = cluster_loss_weight_after_warmup



        
        # Entraînement
        train_loss, mae_loss, cluster_loss, n_unique_train = train_epoch_merl(
            model, train_loader, optimizer, device,
            mae_loss_weight_init, cluster_loss_weight_init
        )
        
        # Logging
        print(f"Train Loss: {train_loss:.6f}")
        print(f"  - MAE Loss: {mae_loss:.6f}")
        print(f"  - Cluster Loss: {cluster_loss:.6f}")
        print(f"  - Classes prédites: {n_unique_train}")

        
        losses_history['total'].append(train_loss)
        losses_history['mae'].append(mae_loss)
        losses_history['cluster'].append(cluster_loss)
        
        
        # TensorBoard - Losses
        writer.add_scalar('Loss/total', train_loss, epoch)
        writer.add_scalar('Loss/mae', mae_loss, epoch)
        writer.add_scalar('Loss/cluster', cluster_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler
        scheduler.step()
        
        # Enregistrer le nombre de clusters
        cluster_evolution.append(model.clustering_layer.n_classes)
        
        # Adaptation des clusters (fusion et séparation)
        if epoch > warmup_epochs and (epoch - warmup_epochs) % adapt_clusters_every == 0:
            print(f"\n{'='*70}")
            print(f"ADAPTING CLUSTERS (Epoch {epoch+1})")
            print(f"{'='*70}")
            
            adapt_stats = adapt_clusters(
                model, train_loader, device,
                merge_threshold=merge_threshold,
                split_threshold=split_threshold,
                n_samples=1000
            )
            
            print(f"  Initial: {adapt_stats['initial_clusters']} clusters")
            print(f"  Final: {adapt_stats['final_clusters']} clusters")
            print(f"  Merges: {adapt_stats['n_merges']}, Splits: {adapt_stats['n_splits']}")
            
            # Mettre à jour l'optimiseur pour inclure les nouveaux paramètres
            optimizer = optim.Adam([
                {'params': model.mae.parameters(), 'lr': lr_after_warmup_mae},
                {'params': model.clustering_layer.parameters(), 'lr': lr_after_warmup_clustering}
            ], weight_decay=weight_decay)
        
        # Visualisation PCA toutes les 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"Generating PCA visualization for epoch {epoch+1}")
            print(f"{'='*70}")
            
            pca_dir = os.path.join(outdir, 'pca_plots')
            
            # PCA sur le train set
            visualize_latent_space_pca(
                model, train_loader, device, 
                epoch=epoch+1, 
                save_dir=pca_dir,
                n_samples=500
            )
            
            # PCA sur le test set si disponible
            if test_loader:
                visualize_latent_space_pca(
                    model, test_loader, device, 
                    epoch=epoch+1, 
                    save_dir=os.path.join(pca_dir, 'test'),
                    n_samples=500
                )
               
            
        # Sauvegarder tous les 20 epochs
        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizer, checkpoint_dir, epoch, best_accuracy)
    
    writer.close()
    
    # Récupérer les prédictions finales
    print("\n" + "="*70)
    print("COLLECTING FINAL PREDICTIONS")
    print("="*70)
    train_preds, train_latents, test_preds, test_latents = get_final_predictions(
        model, train_loader, test_loader, device
    )
    
    print(f"Train predictions shape: {train_preds.shape}")
    if test_preds is not None:
        print(f"Test predictions shape: {test_preds.shape}")
    print(f"Unique clusters in train: {len(np.unique(train_preds))}")
    if test_preds is not None:
        print(f"Unique clusters in test: {len(np.unique(test_preds))}")
    
    # Sauvegarder les prédictions
    predictions_path = os.path.join(outdir, 'final_predictions.npz')
    np.savez(
        predictions_path,
        train_predictions=train_preds,
        train_latents=train_latents,
        test_predictions=test_preds,
        test_latents=test_latents,
        train_materials=train_mats,
        test_materials=test_mats
    )
    print(f"✓ Final predictions saved: {predictions_path}")
    
    # Créer un fichier texte avec un résumé des clusters
    summary_path = os.path.join(outdir, 'clustering_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CLUSTERING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("TRAIN SET CLUSTERING:\n")
        f.write("-" * 70 + "\n")
        for cluster_id in np.unique(train_preds):
            mask = train_preds == cluster_id
            materials = [train_mats[i] for i in range(len(train_mats)) if i < len(train_preds) and mask[i]]
            f.write(f"Cluster {cluster_id}: {np.sum(mask)} samples\n")
            if materials:
                f.write(f"  Materials: {materials}\n")
        
        if test_preds is not None:
            f.write("\n\nTEST SET CLUSTERING:\n")
            f.write("-" * 70 + "\n")
            for cluster_id in np.unique(test_preds):
                mask = test_preds == cluster_id
                materials = [test_mats[i] for i in range(len(test_mats)) if i < len(test_preds) and mask[i]]
                f.write(f"Cluster {cluster_id}: {np.sum(mask)} samples\n")
                if materials:
                    f.write(f"  Materials: {materials}\n")
    
    print(f"✓ Clustering summary saved: {summary_path}")
    
    # Sauvegarde finale
    final_model_path = os.path.join(outdir, 'mae_clustering_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Résultats finaux
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    
    # Sauvegarder les infos d'entraînement
    train_info = {
        'train_mats': train_mats,
        'test_mats': test_mats,
        'best_accuracy': best_accuracy,
        'losses_history': losses_history,
        'cluster_evolution': cluster_evolution,  # Ajout de l'évolution des clusters
        'model_config': {
            'dim_latent': dim_latent,
            'mask_ratio': mask_ratio,
            'warmup_epochs': warmup_epochs,
            'merge_threshold': merge_threshold,
            'split_threshold': split_threshold,
            'adapt_clusters_every': adapt_clusters_every,
        }
    }
    
    train_info_path = os.path.join(outdir, 'mae_clustering_train_info.npy')
    np.save(train_info_path, train_info)
    print(f"✓ Training info saved: {train_info_path}")
    
    print("="*70)













