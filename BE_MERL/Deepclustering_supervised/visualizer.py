import os
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch






def visualize_latent_space_pca(model, data_loader, device, epoch, save_dir='results/pca_plots', n_samples=1000):
    """
    Visualise l'espace latent en 2D avec PCA et colore par classe prédite.
    Affiche une fenêtre matplotlib avec les noms des matériaux visibles.
    
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
            mats = batch_data['mat']
            
            mae_loss, out, mask, latent, q = model(X)
            
            # Prédictions
            pred = torch.argmax(q, dim=1)
            
            all_latents.append(latent.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_mats.extend(mats)
            
            if len(all_latents) * X.shape[0] >= n_samples:
                break
    
    # Concatener
    latents = np.concatenate(all_latents, axis=0)[:n_samples]
    preds = np.concatenate(all_preds, axis=0)[:n_samples]
    mats = all_mats[:n_samples]
    
    # PCA
    print(f"\nComputing PCA for epoch {epoch}...")
    
    # Vérifier qu'on a assez d'échantillons pour PCA
    n_pca_components = min(2, latents.shape[0] - 1)
    if latents.shape[0] < 2:
        print(f"Warning: Only {latents.shape[0]} samples available, skipping PCA visualization")
        return None, None
    
    pca = PCA(n_components=n_pca_components)
    latents_2d = pca.fit_transform(latents)
    
    # ============ VISUALISATION MATPLOTLIB ============
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Scatter plot coloré par classe prédite
    scatter = ax.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=preds,
        cmap='tab20',
        alpha=0.7,
        s=120,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Afficher les noms des matériaux directement sur les points
    for x, y, mat in zip(latents_2d[:, 0], latents_2d[:, 1], mats):
        label = mat.split('/')[-1] if '/' in mat else mat
        label = label.replace('.binary', '')
        ax.annotate(
            label,
            (x, y),
            fontsize=8,
            alpha=0.8,
            ha='center',
            va='bottom'
        )
    
    ax.set_title(
        f'PCA - Epoch {epoch} - Classes Prédites\n{len(np.unique(preds))} clusters détectés',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', rotation=270, labelpad=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder version matplotlib
    save_path = os.path.join(save_dir, f'pca_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Static PCA plot saved: {save_path}")
    
    # Afficher la fenêtre matplotlib (bloquante jusqu'à fermeture)
    print(f"📊 Displaying PCA plot for epoch {epoch}...")
    #plt.show()
    
    return latents_2d, preds


def visualize_mae_latent_space_supervised(model, data_loader, device, epoch, save_dir='results/pca_plots', 
                                          n_samples=1000, show_plot=False, use_3d=True):
    """
    Visualise l'espace latent du MAE en 2D ou 3D avec PCA et colore par classe réelle (labels supervisés).
    Extrait le CLS token de chaque échantillon comme représentation.
    
    Args:
        model: Modèle MAE entraîné
        data_loader: DataLoader (train ou test)
        device: Device (cuda ou cpu)
        epoch: Numéro d'epoch actuel
        save_dir: Répertoire pour sauvegarder les plots
        n_samples: Nombre maximum d'échantillons à visualiser
        show_plot: Si True, affiche le plot avec plt.show()
        use_3d: Si True, génère aussi un plot 3D avec les 3 premières composantes
    """
    from merlDB.database import LABELS_MAT
    from mpl_toolkits.mplot3d import Axes3D
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    all_cls_tokens = []
    all_labels = []
    all_mats = []
    
    print(f"\nExtracting latent representations for epoch {epoch}...")
    
    with torch.no_grad():
        for batch_data in data_loader:
            X = batch_data['values'].to(device)
            mats = batch_data['mat']
            labels = batch_data['label'].cpu().numpy()
            
            # Forward pass pour obtenir les latents
            _, _, _, latent = model(X, mask_ratio=0.0, ispermute=False, issupervised=False)
            
            # Extraire le CLS token (première position) - représentation globale
            cls_token = latent[:, 0, :].cpu().numpy()  # (batch_size, embed_dim)
            
            all_cls_tokens.append(cls_token)
            all_labels.extend(labels)
            all_mats.extend(mats)
            
            if len(all_cls_tokens) * X.shape[0] >= n_samples:
                break
    
    # Concatener
    cls_tokens = np.concatenate(all_cls_tokens, axis=0)[:n_samples]
    labels = np.array(all_labels)[:n_samples]
    mats = all_mats[:n_samples]
    
    print(f"CLS tokens shape: {cls_tokens.shape}")
    print(f"Number of samples: {len(labels)}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # PCA
    print(f"Computing PCA...")
    
    if cls_tokens.shape[0] < 2:
        print(f"Warning: Only {cls_tokens.shape[0]} samples available, skipping PCA visualization")
        return None
    
    # Calculer 3 composantes si 3D demandé, sinon 2
    n_components = 3 if use_3d else 2
    n_components = min(n_components, cls_tokens.shape[0] - 1, cls_tokens.shape[1])
    
    pca = PCA(n_components=n_components)
    latents_pca = pca.fit_transform(cls_tokens)
    
    # Créer une palette de couleurs pour chaque classe
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # ============ VISUALISATION 2D ============
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Scatter plot coloré par classe réelle
    for i, label_id in enumerate(unique_labels):
        mask = labels == label_id
        ax.scatter(
            latents_pca[mask, 0],
            latents_pca[mask, 1],
            c=[colors[i]],
            label=f'{LABELS_MAT[label_id]} ({label_id})',
            alpha=0.7,
            s=120,
            edgecolors='black',
            linewidth=0.8
        )
    
    # Afficher les noms des matériaux sur certains points (pas tous pour éviter le fouillis)
    # Afficher 1 point sur 3 maximum
    step = max(1, len(mats) // 30)  # Maximum ~30 labels
    for idx in range(0, len(mats), step):
        x, y = latents_pca[idx, 0], latents_pca[idx, 1]
        mat = mats[idx]
        label = mat.split('/')[-1] if '/' in mat else mat
        label = label.replace('.binary', '')
        ax.annotate(
            label,
            (x, y),
            fontsize=7,
            alpha=0.6,
            ha='center',
            va='bottom'
        )
    
    ax.set_title(
        f'PCA 2D - Espace Latent MAE (CLS Token) - Epoch {epoch}\n'
        f'Coloré par Classes Réelles ({len(unique_labels)} classes)',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    
    # Légende avec les noms de classes
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        fontsize=9,
        framealpha=0.9,
        title='Classes de Matériaux'
    )
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder 2D
    save_path_2d = os.path.join(save_dir, f'pca_2d_supervised_epoch_{epoch:03d}.png')
    plt.savefig(save_path_2d, dpi=150, bbox_inches='tight')
    print(f"✓ PCA 2D plot saved: {save_path_2d}")
    
    # Afficher 2D si demandé
    if show_plot:
        print(f"📊 Displaying 2D PCA plot...")
        plt.show()
    else:
        plt.close()
    
    # ============ VISUALISATION 3D (si demandé) ============
    if use_3d and n_components >= 3:
        print(f"Generating 3D PCA visualization...")
        
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot 3D coloré par classe réelle
        for i, label_id in enumerate(unique_labels):
            mask = labels == label_id
            ax.scatter(
                latents_pca[mask, 0],
                latents_pca[mask, 1],
                latents_pca[mask, 2],
                c=[colors[i]],
                label=f'{LABELS_MAT[label_id]} ({label_id})',
                alpha=0.7,
                s=120,
                edgecolors='black',
                linewidth=0.8
            )
        
        # Afficher quelques noms de matériaux en 3D
        step = max(1, len(mats) // 20)  # Moins de labels en 3D pour éviter l'encombrement
        for idx in range(0, len(mats), step):
            x, y, z = latents_pca[idx, 0], latents_pca[idx, 1], latents_pca[idx, 2]
            mat = mats[idx]
            label = mat.split('/')[-1] if '/' in mat else mat
            label = label.replace('.binary', '')
            ax.text(x, y, z, label, fontsize=6, alpha=0.5)
        
        ax.set_title(
            f'PCA 3D - Espace Latent MAE (CLS Token) - Epoch {epoch}\n'
            f'Coloré par Classes Réelles ({len(unique_labels)} classes)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', labelpad=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', labelpad=10)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', labelpad=10)
        
        # Légende
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            ncol=1,
            fontsize=8,
            framealpha=0.9,
            title='Classes de Matériaux'
        )
        
        # Rotation pour meilleure vue
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Sauvegarder 3D
        save_path_3d = os.path.join(save_dir, f'pca_3d_supervised_epoch_{epoch:03d}.png')
        plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
        print(f"✓ PCA 3D plot saved: {save_path_3d}")
        
        # Afficher 3D si demandé
        if show_plot:
            print(f"📊 Displaying 3D PCA plot...")
            plt.show()
        else:
            plt.close()
    
    return latents_pca, labels