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

from visualizer import visualize_latent_space_pca, visualize_mae_latent_space_supervised
from models import MAEWithClustering
import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import math
import sys
from merlDB.database import LABELS_MAT
try:
    from mae_model_brdf import MaskedAutoencoderViT3D
except ImportError:
    print("Warning: mae_model_brdf not found")
    MaskedAutoencoderViT3D = None



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







def train_epoch_merl(model, data_loader, optimizer, device, epoch, loss_scaler, 
                     log_writer=None, args=None):
    """
    Entraîne le modèle MAE avec supervision pour une epoch
    
    Args:
        model: Modèle MAE à entraîner
        data_loader: DataLoader pour les données d'entraînement
        optimizer: Optimiseur
        device: Device (cuda ou cpu)
        epoch: Numéro d'epoch actuel
        loss_scaler: Loss scaler pour mixed precision training
        log_writer: TensorBoard writer (optionnel)
        args: Arguments d'entraînement
    
    Returns:
        dict: Statistiques moyennes de l'epoch
        float: Valeur de loss réduite
    """
    model.train(True)
    
    # Metric logger pour suivre les métriques
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mae', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_supervised', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    accum_iter = args.accum_iter if hasattr(args, 'accum_iter') else 1
    weight_supervised = args.weight_supervised if hasattr(args, 'weight_supervised') else 0.3
    
    optimizer.zero_grad()
    
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # Ajuster le learning rate si nécessaire
        if data_iter_step % accum_iter == 0:
            if hasattr(args, 'lr') and hasattr(args, 'min_lr'):
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # Récupérer les données
        samples = batch_data['values'].to(device, non_blocking=True)
        labels = batch_data['label'].to(device, non_blocking=True)
        
        # Forward pass avec mixed precision
        with torch.cuda.amp.autocast():
            loss_mae, loss_supervised, _, _, latent = model(
                samples, 
                mask_ratio=args.mask_ratio if hasattr(args, 'mask_ratio') else 0.5,
                ispermute=False,
                issupervised=True,
                labels=labels
            )
            
            # Loss totale
            loss = loss_mae + weight_supervised * loss_supervised
        
        loss_value = loss.item()
        loss_mae_value = loss_mae.item()
        loss_supervised_value = loss_supervised.item()
        
        # Vérifier si la loss est finie
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(f"MAE Loss: {loss_mae_value}, Supervised Loss: {loss_supervised_value}")
            sys.exit(1)
        
        # Gradient accumulation
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        # Mettre à jour les métriques
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_mae=loss_mae_value)
        metric_logger.update(loss_supervised=loss_supervised_value)
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        # TensorBoard logging
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # epoch_1000x pour calibrer les courbes selon le batch size
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_mae', loss_mae_value, epoch_1000x)
            log_writer.add_scalar('train_loss_supervised', loss_supervised_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    
    # Synchroniser les statistiques entre tous les processus (si distribué)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, loss_value_reduce




def get_final_prediction(model, test_dataset, device):
    """retourne le score de prédiction finale pour l'ensemble de test"""

    good_predictions = 0
    total_predictions = 0

    labels_good_pred = [0 for i in range(len(LABELS_MAT))]
    labels_total = [0 for i in range(len(LABELS_MAT))]

    for batch_data in test_dataset:
        X = batch_data['values'].to(device)
        labels = batch_data['label'].to(device)
        
        with torch.no_grad():
            classe = model.forward_clustering(X)
           
            good_predictions += (classe == labels).sum().item()
            
            # Mettre à jour les compteurs pour chaque classe
            for i in range(len(labels)):
                label = labels[i].item()
                labels_total[label] += 1
                if classe[i].item() == label:
                    labels_good_pred[label] += 1

            total_predictions += labels.size(0)

    accuracy = good_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, labels_good_pred, labels_total

        
        


if __name__ == "__main__":
      
    # Paramètres du modèle MAE
    dim_latent = 12#768                # Dimension de l'espace latent
    mask_ratio = 0.5                # Ratio de masquage
    poids = 3                        # Poids pour la perte MAE
    norme = 0#2                        # Norme de la fonction de perte (1 ou 2)
    
    # Paramètres d'entraînement
    epochs = 1000                     # Nombre d'epochs
    batch_size = 8                  # Batch size AUGMENTÉ de 4 à 8
    lr_init_mae = 1e-3                      # Learning rate CORRIGÉ de 1e-1 à 1e-4
    weight_decay = 0.            # Weight decay
    accum_iter = 1                   # Gradient accumulation iterations
    weight_supervised = 0.3          # Poids de la loss supervisée
    

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
    print("MAE + SUPERVISED TRAINING ON MERL")
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
    data = load_merl_datasets(args=args_data, device=device,classes_dict=LABELS_MAT)
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
        norme=norme,
        smoothness_weight=0.1
    )
    mae.to(device)
    

    print(f"MAE Dimension: {dim_latent}")
    print(f"Mask Ratio: {mask_ratio}")
    print(f"Model parameters: {sum(p.numel() for p in mae.parameters()):,}")
    
    # Créer un objet args pour les paramètres d'entraînement
    class TrainArgs:
        def __init__(self):
            self.mask_ratio = mask_ratio
            self.accum_iter = accum_iter
            self.weight_supervised = weight_supervised
            self.lr = lr_init_mae
            self.min_lr = lr_init_mae / 100  # LR minimum pour le scheduler
            self.warmup_epochs = 0  # Pas de warmup
            self.epochs = epochs
    
    train_args = TrainArgs()
    
    # Loss scaler pour mixed precision training
    loss_scaler = NativeScaler()
    
    # Optimiseur et scheduler
    optimizer = optim.Adam([
        {'params': mae.parameters(), 'lr': lr_init_mae},
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

        # Entraînement
        train_stats, avg_loss = train_epoch_merl(
            mae, train_loader, optimizer, device, epoch, 
            loss_scaler, log_writer=writer, args=train_args
        )
        
        # Logging
        print(f"Train Loss: {avg_loss:.6f}")
        if 'loss_mae' in train_stats:
            print(f"  - MAE Loss: {train_stats['loss_mae']:.6f}")
        if 'loss_supervised' in train_stats:
            print(f"  - Supervised Loss: {train_stats['loss_supervised']:.6f}")
        
        losses_history['total'].append(avg_loss)
        if 'loss_mae' in train_stats:
            losses_history['mae'].append(train_stats['loss_mae'])
        if 'loss_supervised' in train_stats:
            losses_history['supervised'].append(train_stats['loss_supervised'])
        
        # TensorBoard - Losses principales déjà loggées dans train_epoch_merl
        writer.add_scalar('Loss/total_epoch', avg_loss, epoch)
        if 'lr' in train_stats:
            writer.add_scalar('Learning_Rate_epoch', train_stats['lr'], epoch)
        
        # Scheduler
        scheduler.step()
        
        # Visualisation PCA toutes les 50 epochs
        if (epoch + 1) % 100 == 0:
            print(f"\n{'='*70}")
            print(f"Generating PCA visualization for epoch {epoch+1}")
            print(f"{'='*70}")
            
            pca_dir = os.path.join(outdir, 'pca_plots')
            
            # PCA sur le train set avec les vraies classes (2D et 3D)
            visualize_mae_latent_space_supervised(
                mae, train_loader, device, 
                epoch=epoch+1, 
                save_dir=pca_dir,
                n_samples=500,
                show_plot=False,
                use_3d=True  # Générer aussi la version 3D
            )
            
            # PCA sur le test set si disponible
            if test_loader:
                visualize_mae_latent_space_supervised(
                    mae, test_loader, device, 
                    epoch=epoch+1, 
                    save_dir=os.path.join(pca_dir, 'test'),
                    n_samples=100,
                    show_plot=False,
                    use_3d=True
                )

        #tester la classification avec le test data
        if (epoch + 1) % 1 == 0 and test_loader is not None:
            print(f"\n{'='*70}")
            print(f"Evaluating classification accuracy for epoch {epoch+1}")
            print(f"{'='*70}")
            
            accuracy, labels_good_pred, labels_total = get_final_prediction(mae, test_loader, device)
            print(f"Test Accuracy: {accuracy:.4f}")
            
            # Afficher les résultats par classe
            print("Accuracy par classe:")
            for i in range(len(labels_total)):
                if labels_total[i] > 0:
                    acc = labels_good_pred[i] / labels_total[i]
                    print(f"  Classe {LABELS_MAT[i]}: {acc:.4f} ({labels_good_pred[i]}/{labels_total[i]})")

            
            
        # Sauvegarder tous les 20 epochs
        if (epoch + 1) % 200 == 0:
            save_checkpoint(mae, optimizer, checkpoint_dir, epoch, best_accuracy)
    
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















