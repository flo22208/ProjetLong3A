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

import math
import sys
try:
    from model import EncoderViT3D
except ImportError:
    print("Warning: model.py not found")
    EncoderViT3D = None



from psnr_result import psnr


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




def train_epoch_disney(model, BRDF, optimizer, device, epoch, args=None):
    """
    Entraîne le modèle Transformer avec supervision pour une epoch
    
    Args:
        model: Modèle Transformer à entraîner
        BRDF: Fonction de BRDF à utiliser
        optimizer: Optimiseur
        device: Device (cuda ou cpu)
        epoch: Numéro d'epoch actuel
        args: Arguments d'entraînement
    
    Returns:
        float: Valeur de loss réduite
    """
    model.train(True)
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    optimizer.zero_grad()
    
    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # TODO : tirer params disney 12 batchs, lancer BRDF et postprocess

        # Forward pass avec mixed precision
        with torch.cuda.amp.autocast():
            loss, pred_params = model(
                samples
            )
        
        loss_value = loss.item()
        
        # Vérifier si la loss est finie
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
    

    return loss




def get_final_prediction(model, BRDF, device):
    """retourne le score de prédiction finale pour un ensemble de test"""

    # good_predictions = 0
    # total_predictions = 0

    # labels_good_pred = [0 for i in range(len(LABELS_MAT))]
    # labels_total = [0 for i in range(len(LABELS_MAT))]

    # for batch_data in test_dataset:
    #     X = batch_data['values'].to(device)
    #     labels = batch_data['label'].to(device)
        
    #     with torch.no_grad():
    #         classe = model.forward_clustering(X)
           
    #         good_predictions += (classe == labels).sum().item()
            
    #         # Mettre à jour les compteurs pour chaque classe
    #         for i in range(len(labels)):
    #             label = labels[i].item()
    #             labels_total[label] += 1
    #             if classe[i].item() == label:
    #                 labels_good_pred[label] += 1

    #         total_predictions += labels.size(0)

    # accuracy = good_predictions / total_predictions if total_predictions > 0 else 0
    # return accuracy, labels_good_pred, labels_total
    return

        
        


if __name__ == "__main__":
      
    # Paramètres du modèle MAE
    dim_latent = 12#768                # Dimension de l'espace latent

    # Paramètres d'entraînement
    epochs = 5                     # Nombre d'epochs
    batch_size = 12                  # Batch size 
    lr = 1e-3                      # Learning rate
    

    # Chemins et répertoires
    outdir = 'results/'       # Répertoire de sortie (warmup approach)
    checkpoint_dir = 'checkpoints/'  # Répertoire pour les checkpoints

    
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
    print("TRAINING ON DISNEY BRDF")
    print("="*70)
    
    # Device
    device = torch.device(device_name)
    print(f"\nDevice: {device}")
    
    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Créer les répertoires
    setup_output_dirs(outdir, checkpoint_dir)
    
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
    
    # Nombre de classes = nombre de catégories définies dans CLASSES
    
    # Créer le modèle Encoder
    print("\n" + "="*70)
    print("CREATING ENCODER MODEL")
    print("="*70)
    
    if EncoderViT3D is None:
        raise ImportError("Encoder model not available")
    
    model = EncoderViT3D(
        embed_dim=dim_latent,
    )
    model.to(device)
    

    print(f"Disney params dimension: {dim_latent}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Créer un objet args pour les paramètres d'entraînement
    class TrainArgs:
        def __init__(self):
            self.lr = lr
            self.epochs = epochs
    
    train_args = TrainArgs()
    
    # Optimiseur et scheduler
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Entraînement
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_accuracy = 0
    losses_history = {
        'total': [],
        'train_acc': [],
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)

        # Entraînement
        avg_loss = train_epoch_disney(
            model, BRDF, optimizer, device, epoch, args=train_args
        )
        
        # Logging
        print(f"Train Loss: {avg_loss:.6f}")

        losses_history['total'].append(avg_loss)
        
        # Scheduler
        scheduler.step()            
            
        # Sauvegarder tous les 20 epochs
        if (epoch + 1) % 200 == 0:
            save_checkpoint(model, optimizer, checkpoint_dir, epoch, best_accuracy)
    
    writer.close()

    # Visualisation finale
    print("\n" + "="*70)
    print("GENERATING FINAL PCA VISUALIZATION")
    print("="*70)
    
    pca_dir = os.path.join(outdir, 'pca_plots')
    
    # PCA finale sur le train set (2D et 3D)
    visualize_mae_latent_space_supervised(
        mae, train_loader, device, 
        epoch=epochs, 
        save_dir=pca_dir,
        n_samples=len(train_dataset),  # Tous les échantillons
        show_plot=True,  # Afficher à la fin
        use_3d=True  # Générer aussi la version 3D
    )
    
    # PCA finale sur le test set si disponible
    if test_loader:
        visualize_mae_latent_space_supervised(
            mae, test_loader, device, 
            epoch=epochs, 
            save_dir=os.path.join(pca_dir, 'test'),
            n_samples=len(test_dataset),
            show_plot=True,
            use_3d=True
        )
    
    print("="*70)

    ## enregistrer les poids du modèle final
    final_model_path = os.path.join(outdir, 'mae_final_supervised.pt')
    torch.save(mae.state_dict(), final_model_path)

    ## visualiation des psnr
    print("\n" + "="*70)