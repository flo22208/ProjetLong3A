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

try:
    from mae_model_brdf import MaskedAutoencoderViT3D
except ImportError:
    print("Warning: mae_model_brdf not found")
    MaskedAutoencoderViT3D = None


class MERLClusteringLayer(nn.Module):
    """
    Couche de clustering pour MERL qui apprend des centroïdes de classes.
    """
    def __init__(self, n_classes=10, latent_dim=768, alpha=1.0):
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
    def __init__(self, mae, n_classes=10, latent_dim=768):
        super(MAEWithClustering, self).__init__()
        self.mae = mae
        self.n_classes = n_classes
        self.expected_latent_dim = latent_dim
        self.clustering = None  # Sera initialisé lors du premier forward
        
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
        
        # Initialiser le clustering layer avec la vraie dimension latente si pas déjà fait
        if self.clustering is None:
            actual_latent_dim = latent.shape[-1]
            print(f"Initializing clustering layer with latent_dim={actual_latent_dim}")
            self.clustering = MERLClusteringLayer(n_classes=self.n_classes, latent_dim=actual_latent_dim)
            self.clustering.to(latent.device)
        
        # Clustering sur les représentations latentes
        q = self.clustering(latent)
        
        return loss, out, mask, latent, q





def target_distribution(q):
    """Calcule la distribution cible pour le clustering"""
    # q shape: (batch_size, n_classes)
    weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-10)  # Éviter division par zéro
    p = weight / (weight.sum(dim=1, keepdim=True) + 1e-10)  # Normaliser par sample
    return p


def get_class_index_from_material(material_name, classes_dict):
    """
    Retourne l'index de classe pour un matériau donné basé sur CLASSES
    """
    for class_idx, (class_name, materials) in enumerate(classes_dict.items()):
        if material_name in materials:
            return class_idx
    return -1  # Matériau non trouvé


def combined_merl_loss(mae_loss, q, class_indices,
                       mae_weight=1.0, cluster_weight=0.5, supervised_weight=1.0):
    """
    Loss combinée: MAE + Clustering + Supervision
    Utilise les vrais indices de classe de CLASSES
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
    
    # Loss supervisée utilisant les vrais indices de classe
    supervised_loss = nn.functional.cross_entropy(q, class_indices)
    
    # Loss totale
    total_loss = (mae_weight * mae_loss_val + 
                  cluster_weight * cluster_loss +
                  supervised_weight * supervised_loss)
    
    return total_loss, mae_loss_val, cluster_loss, supervised_loss


def train_epoch_merl(model, train_loader, optimizer, device, mae_loss_weight, cluster_loss_weight, supervised_loss_weight):
    """Entraîne le modèle MAE+Clustering pour une epoch"""
    model.train()
    
    total_loss = 0
    total_mae_loss = 0
    total_cluster_loss = 0
    total_supervised_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_data in pbar:
        X = batch_data['values'].to(device)
        class_indices = batch_data['class_idx'].to(device)
        
        # Forward pass
        mae_loss, out, mask, latent, q = model(X)
        
        # Calcul de la loss combinée avec les vrais indices de classe
        loss, mae_l, cluster_l, supervised_l = combined_merl_loss(
            mae_loss, q, class_indices,
            mae_loss_weight,
            cluster_loss_weight,
            supervised_loss_weight
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        total_loss += loss.item()
        total_mae_loss += mae_l.item() if isinstance(mae_l, torch.Tensor) else mae_l
        total_cluster_loss += cluster_l.item()
        total_supervised_loss += supervised_l.item()
        
        # Précision avec les vrais indices de classe
        pred = torch.argmax(q, dim=1)
        correct += (pred == class_indices).sum().item()
        total += class_indices.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_mae_loss = total_mae_loss / len(train_loader)
    avg_cluster_loss = total_cluster_loss / len(train_loader)
    avg_supervised_loss = total_supervised_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    n_unique_preds = len(np.unique(all_preds))
    
    return avg_loss, avg_mae_loss, avg_cluster_loss, avg_supervised_loss, accuracy, n_unique_preds


def evaluate_merl(model, test_loader, device):
    """Évalue le modèle sur le dataset de test"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_latents = []
    all_mats = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc='Evaluating'):
            X = batch_data['values'].to(device)
            class_indices = batch_data['class_idx'].to(device)
            mats = batch_data['mat']
            
            mae_loss, out, mask, latent, q = model(X)
            
            pred = torch.argmax(q, dim=1)
            correct += (pred == class_indices).sum().item()
            total += class_indices.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(class_indices.cpu().numpy())
            all_latents.append(latent.cpu().detach())
            all_mats.extend(mats)
    
    accuracy = 100. * correct / total
    all_latents = torch.cat(all_latents, dim=0)
    
    n_unique_preds = len(np.unique(all_preds))
    
    return accuracy, np.array(all_preds), np.array(all_labels), all_latents, all_mats, n_unique_preds


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


def visualize_latent_space_pca(model, data_loader, device, epoch, save_dir='results/pca_plots', n_samples=1000):
    """
    Visualise l'espace latent en 2D avec PCA et colore par classe prédite et vraie classe.
    
    Args:
        model: Modèle entraîné
        data_loader: DataLoader (train ou test)
        device: Device (cuda ou cpu)
        epoch: Numéro d'epoch actuel
        save_dir: Répertoire pour sauvegarder les plots
        n_samples: Nombre maximum d'échantillons à visualiser
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    all_latents = []
    all_labels = []
    all_preds = []
    all_mats = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            X = batch_data['values'].to(device)
            class_indices = batch_data['class_idx'].to(device)
            mats = batch_data['mat']
            
            mae_loss, out, mask, latent, q = model(X)
            
            # Prédictions
            pred = torch.argmax(q, dim=1)
            
            all_latents.append(latent.cpu().numpy())
            all_labels.append(class_indices.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_mats.extend(mats)
            
            if len(all_latents) * X.shape[0] >= n_samples:
                break
    
    # Concatener
    latents = np.concatenate(all_latents, axis=0)[:n_samples]
    labels = np.concatenate(all_labels, axis=0)[:n_samples]
    preds = np.concatenate(all_preds, axis=0)[:n_samples]
    mats = all_mats[:n_samples]
    
    # PCA
    print(f"\nComputing PCA for epoch {epoch}...")
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Créer la figure avec 2 sous-plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Coloré par vraie classe
    scatter1 = ax1.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                           c=labels, cmap='tab10', alpha=0.6, s=20)
    ax1.set_title(f'PCA - Epoch {epoch} - Vraies Classes', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Classe', rotation=270, labelpad=15)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coloré par classe prédite
    scatter2 = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                           c=preds, cmap='tab10', alpha=0.6, s=20)
    ax2.set_title(f'PCA - Epoch {epoch} - Classes Prédites', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Cluster', rotation=270, labelpad=15)
    ax2.grid(True, alpha=0.3)
    
    # Calculer l'accuracy
    accuracy = 100.0 * (preds == labels).sum() / len(labels)
    
    plt.suptitle(f'Visualisation PCA de l\'espace latent - Accuracy: {accuracy:.2f}%',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Sauvegarder
    save_path = os.path.join(save_dir, f'pca_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ PCA plot saved: {save_path}")
    
    return latents_2d, labels, preds


def init_centroids_from_representative_materials(model, data_loader, device, classes_dict, max_batches=10):
    """
    Initialise les centroïdes en prenant un matériau représentatif par classe (CLASSES).
    """
    model.eval()
    # forcer l'initialisation de la couche clustering
    with torch.no_grad():
        for batch_data in data_loader:
            X = batch_data['values'].to(device)
            model(X)
            break

    class_names = list(classes_dict.keys())
    class_to_mat = {i: classes_dict[name][0] for i, name in enumerate(class_names) if classes_dict[name]}

    centroids = model.clustering.centroids.data.clone()
    for class_idx, mat_name in class_to_mat.items():
        latents_list = []
        with torch.no_grad():
            for b_idx, batch_data in enumerate(data_loader):
                X = batch_data['values'].to(device)
                mats = batch_data['mat']
                _, _, _, latent, _ = model(X)
                # sélectionner les samples du matériau cible
                mask = [m == mat_name for m in mats]
                if any(mask):
                    latents_list.append(latent[torch.tensor(mask, device=latent.device)].detach())
                if b_idx + 1 >= max_batches:
                    break
        if latents_list:
            class_latents = torch.cat(latents_list, dim=0)
            centroids[class_idx] = class_latents.mean(dim=0)

    model.clustering.centroids.data.copy_(centroids)
    print("✓ Centroids initialized from representative materials")


def train_mae_clustering_merl():
    """Entraînement MAE + Clustering sur MERL"""
    
    CLASSES = {
        'metal': [
            'alum-bronze',
            'aluminium',
            'alumina-oxide',
            'brass',
            'black-oxidized-steel',
            'chrome',
            'chrome-steel',
            'grease-covered-steel',
            'nickel',
            'steel',
            'ss440',
            'tungsten-carbide'
        ],
        'plastic': [
            'black-soft-plastic',
            'gray-plastic',
            'green-plastic',
            'maroon-plastic',
            'pink-plastic',
            'red-plastic',
            'yellow-plastic',
            'yellow-matte-plastic',
            'red-specular-plastic',
            'delrin',
            'nylon',
            'polyethylene',
            'polyurethane-foam',
            'pvc',
            'teflon',
            'black-phenolic',
            'red-phenolic',
            'yellow-phenolic',
            'specular-black-phenolic',
            'specular-blue-phenolic',
            'specular-green-phenolic',
            'specular-maroon-phenolic',
            'specular-orange-phenolic',
            'specular-red-phenolic',
            'specular-violet-phenolic',
            'specular-white-phenolic',
            'specular-yellow-phenolic'
        ],
        'paint': [
            'color-changing-paint1',
            'color-changing-paint2',
            'color-changing-paint3',
            'dark-blue-paint',
            'dark-red-paint',
            'light-red-paint',
            'orange-paint',
            'purple-paint',
            'yellow-paint',
            'white-paint',
            'pearl-paint',
            'blue-metallic-paint',
            'blue-metallic-paint2',
            'gold-metallic-paint',
            'gold-metallic-paint2',
            'gold-metallic-paint3',
            'gold-paint',
            'silver-metallic-paint',
            'silver-metallic-paint2',
            'silver-paint',
            'green-metallic-paint',
            'green-metallic-paint2',
            'red-metallic-paint'
        ],
        'fabric': [
            'beige-fabric',
            'black-fabric',
            'blue-fabric',
            'green-fabric',
            'light-brown-fabric',
            'pink-fabric',
            'pink-fabric2',
            'pink-felt',
            'red-fabric',
            'red-fabric2',
            'white-fabric',
            'white-fabric2',
            'dark-specular-fabric'
        ]
    }


    
    # ============== CONFIGURATION À PERSONNALISER ==============
    
    # Paramètres du modèle MAE
    dim_latent = 768                # Dimension de l'espace latent
    mask_ratio = 0.5                # Ratio de masquage
    poids = 2                        # Poids pour la perte MAE
    norme = 2                        # Norme de la fonction de perte (1 ou 2)
    
    # Paramètres d'entraînement
    epochs = 200                     # Nombre d'epochs
    batch_size = 8                   # Batch size AUGMENTÉ de 4 à 8
    lr = 1e-4                        # Learning rate CORRIGÉ de 1e-1 à 1e-4
    weight_decay = 0.            # Weight decay
    
    # Démarrer le clustering après N epochs
    warmup_epochs = 15
    
    # Poids des losses - RÉÉQUILIBRÉ
    mae_loss_weight = 1.0            # Poids de la loss MAE
    cluster_loss_weight = 0#0.5        # Poids de la loss clustering RÉDUIT de 2.0 à 1.0
    supervised_loss_weight = 0#1.0     # Poids de la loss supervisée RÉDUIT de 2.0 à 1.0
    
    # Chemins et répertoires
    outdir = 'results/'              # Répertoire de sortie
    checkpoint_dir = 'checkpoints/'  # Répertoire pour les checkpoints

    
    
    # Données MERL
    merldir = 'merlDB/db/brdfs/'     # Chemin vers la base de données MERL
    mediandir = 'merlDB/db/'         # Chemin vers le fichier median MERL
    train_size = 100                 # Nombre de matériaux pour l'entraînement
    monomat = None                   # Si spécifié, charge uniquement ce matériau
    
    # Autres paramètres
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device (cuda ou cpu)
    seed = 0                         # Graine aléatoire
    
    # ============================================================
    
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
    data = load_merl_datasets(args=args_data, device=device, classes_dict=CLASSES)
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_mats = data['train_mats']
    test_mats = data['test_mats']
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")
    
    # Nombre de classes = nombre de catégories définies dans CLASSES
    n_classes = len(CLASSES)
    print(f"Number of material classes: {n_classes}")
    
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
    model = MAEWithClustering(mae, n_classes=n_classes, latent_dim=dim_latent)
    model.to(device)
    
    print(f"MAE Dimension: {dim_latent}")
    print(f"Mask Ratio: {mask_ratio}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimiseur et scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # DataLoaders
    def custom_collate_fn(batch):
        mats = [item['mat'] for item in batch]
        # Mapper chaque matériau à son index de classe basé sur CLASSES
        class_indices = torch.tensor([get_class_index_from_material(item['mat'], CLASSES) for item in batch])
        values = torch.stack([item['values'] for item in batch])
        return {'mat': mats, 'class_idx': class_indices, 'values': values}
    
    train_loader = torchdata.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = None
    if test_dataset:
        test_loader = torchdata.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
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
        
        # Activer clustering après warmup
        use_clustering = (epoch >= warmup_epochs)
        effective_cluster_weight = cluster_loss_weight if use_clustering else 0.0
        
        # Init des centroïdes au début du clustering
        if epoch == warmup_epochs:
            init_centroids_from_representative_materials(
                model, train_loader, device, CLASSES, max_batches=10
            )
            cluster_loss_weight = 1
            supervised_loss_weight = 1
            mae_loss_weight = 0.3
            print(f"✓ Clustering initialized at epoch {epoch+1}")
        
        # Entraînement
        train_loss, mae_loss, cluster_loss, supervised_loss, train_acc, n_unique_train = train_epoch_merl(
            model, train_loader, optimizer, device,
            mae_loss_weight, effective_cluster_weight, supervised_loss_weight
        )
        
        # Logging
        print(f"Train Loss: {train_loss:.6f}")
        print(f"  - MAE Loss: {mae_loss:.6f}")
        print(f"  - Cluster Loss: {cluster_loss:.6f}")
        print(f"  - Supervised Loss: {supervised_loss:.6f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Number of unique predicted classes (train): {n_unique_train}/{n_classes}")
        
        losses_history['total'].append(train_loss)
        losses_history['mae'].append(mae_loss)
        losses_history['cluster'].append(cluster_loss)
        losses_history['supervised'].append(supervised_loss)
        losses_history['train_acc'].append(train_acc)
        
        # Validation
        if test_loader:
            test_acc, test_preds, test_labels, test_latents, test_mats_list, n_unique_test = evaluate_merl(
                model, test_loader, device
            )
            print(f"Test Accuracy: {test_acc:.2f}%")
            print(f"Number of unique predicted classes (test): {n_unique_test}/{n_classes}")
            losses_history['test_acc'].append(test_acc)
            
            # TensorBoard
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            
            # Sauvegarder si meilleure accuracy
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_path = os.path.join(outdir, 'best_model_mae_clustering.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"✓ Best model saved with test accuracy: {test_acc:.2f}%")
        else:
            # Utiliser train accuracy si pas de test set
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                best_model_path = os.path.join(outdir, 'best_model_mae_clustering.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"✓ Best model saved with train accuracy: {train_acc:.2f}%")
        
        # TensorBoard - Losses
        writer.add_scalar('Loss/total', train_loss, epoch)
        writer.add_scalar('Loss/mae', mae_loss, epoch)
        writer.add_scalar('Loss/cluster', cluster_loss, epoch)
        writer.add_scalar('Loss/supervised', supervised_loss, epoch)
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
    
    writer.close()
    
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
            'n_classes': n_classes
        }
    }
    
    train_info_path = os.path.join(outdir, 'mae_clustering_train_info.npy')
    np.save(train_info_path, train_info)
    print(f"✓ Training info saved: {train_info_path}")
    
    print("="*70)


if __name__ == "__main__":
    train_mae_clustering_merl()


