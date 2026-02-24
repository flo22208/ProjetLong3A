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








if __name__ == "__main__":
      
    # Paramètres du modèle MAE
    dim_latent = 768#768                # Dimension de l'espace latent
    mask_ratio = 0.5                # Ratio de masquage
    poids = 2                        # Poids pour la perte MAE
    norme = 2                        # Norme de la fonction de perte (1 ou 2)
    
    # Paramètres d'entraînement
    epochs = 200                     # Nombre d'epochs
    batch_size = 8  
    lr_mae = 1e-4
    lr_clustering = 0                # Batch size AUGMENTÉ de 4 à 8
    lr_init_mae_mae_epoch = 1e-4                        # Learning rate CORRIGÉ de 1e-1 à 1e-4
    lr_init_clustering_mae_epoch = 0               # Learning rate pour le clustering
    lr_after_mae_cluster_epoch = 0          # Learning rate après warmup pour MAE
    lr_after_clustering_cluster_epoch = 1e-3   # Learning rate après warmup pour clustering
    weight_decay = 0.            # Weight decay
    
    epoch_alternate = 2                   # Nombre d'epochs avant d'alterner les poids des losses
    
    # Poids des losses - RÉÉQUILIBRÉ
    mae_loss_weight = 1.0    
    cluster_loss_weight = 0.3                
    
    # Chemins et répertoires
    outdir = 'results_alternate/'    # Répertoire de sortie (alternate approach)
    checkpoint_dir = 'checkpoints_alternate/'  # Répertoire pour les checkpoints

    
    
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
    model = MAEWithClustering(mae, latent_dim=dim_latent,n_classes=10) 
    model.to(device)
    
    print(f"MAE Dimension: {dim_latent}")
    print(f"Mask Ratio: {mask_ratio}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimiseur et scheduler
    optimizer = optim.Adam([
        {'params': model.mae.parameters(), 'lr': lr_mae},
        {'params': model.clustering_layer.parameters(), 'lr': lr_clustering}
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

   
    

    
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)

        # Init des centroïdes avec K-means au début du clustering
        if epoch % epoch_alternate == 0 :
            lr_mae = lr_init_mae_mae_epoch
            lr_clustering = lr_init_clustering_mae_epoch
        else:
            lr_mae = lr_after_mae_cluster_epoch
            lr_clustering = lr_after_clustering_cluster_epoch


        optimizer.param_groups[0]['lr'] = lr_mae
        optimizer.param_groups[1]['lr'] = lr_clustering



        # Entraînement
        train_loss, mae_loss, cluster_loss, n_unique_train = train_epoch_merl(
            model, train_loader, optimizer, device,
            mae_loss_weight, cluster_loss_weight
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
        
        #print(f"\n{'='*70}")
        #print(f"Initializing centroids with K-means before starting clustering")
        #print(f"{'='*70}")
        #model.initialize_centroids_kmeans(train_loader, device, n_samples=1000)
        if cluster_loss<0.001:
            print(f"\n{'='*70}")
            print(f"Initializing centroids with K-means before starting clustering")
            print(f"{'='*70}")
            model.initialize_centroids_kmeans(train_loader, device, n_samples=1000)
    
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
        'model_config': {
            'dim_latent': dim_latent,
            'mask_ratio': mask_ratio,
        }
    }
    
    train_info_path = os.path.join(outdir, 'mae_clustering_train_info.npy')
    np.save(train_info_path, train_info)
    print(f"✓ Training info saved: {train_info_path}")
    
    print("="*70)













