"""
Script pour visualiser l'évolution du nombre de clusters pendant l'entraînement
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cluster_evolution(train_info_path, output_path=None):
    """
    Affiche l'évolution du nombre de clusters au cours de l'entraînement
    
    Args:
        train_info_path: Chemin vers le fichier .npy contenant les infos d'entraînement
        output_path: Chemin optionnel pour sauvegarder le graphique
    """
    # Charger les données
    train_info = np.load(train_info_path, allow_pickle=True).item()
    
    cluster_evolution = train_info.get('cluster_evolution', None)
    losses_history = train_info['losses_history']
    model_config = train_info.get('model_config', {})
    
    if cluster_evolution is None:
        print("Erreur: 'cluster_evolution' non trouvé dans train_info")
        return
    
    epochs = list(range(1, len(cluster_evolution) + 1))
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Évolution du nombre de clusters
    ax1.plot(epochs, cluster_evolution, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=cluster_evolution[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax1.axhline(y=cluster_evolution[-1], color='red', linestyle='--', alpha=0.5, label='Final')
    
    # Marquer les changements significatifs
    changes = []
    for i in range(1, len(cluster_evolution)):
        if cluster_evolution[i] != cluster_evolution[i-1]:
            changes.append(i)
            change = cluster_evolution[i] - cluster_evolution[i-1]
            color = 'green' if change > 0 else 'orange'
            symbol = '↑' if change > 0 else '↓'
            ax1.scatter(i+1, cluster_evolution[i], color=color, s=100, zorder=5, 
                       marker='^' if change > 0 else 'v')
    
    warmup = model_config.get('warmup_epochs', 0)
    if warmup > 0:
        ax1.axvline(x=warmup, color='purple', linestyle=':', linewidth=2, alpha=0.7, label='Warmup end')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_title('Cluster Evolution During Training', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Ajouter des annotations
    merge_threshold = model_config.get('merge_threshold', 'N/A')
    split_threshold = model_config.get('split_threshold', 'N/A')
    adapt_every = model_config.get('adapt_clusters_every', 'N/A')
    
    info_text = f'Merge threshold: {merge_threshold}\n'
    info_text += f'Split threshold: {split_threshold}\n'
    info_text += f'Adapt every: {adapt_every} epochs'
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Losses
    ax2.plot(epochs, losses_history['total'], label='Total Loss', linewidth=2)
    ax2.plot(epochs, losses_history['mae'], label='MAE Loss', linewidth=2)
    ax2.plot(epochs, losses_history['cluster'], label='Cluster Loss', linewidth=2)
    
    if warmup > 0:
        ax2.axvline(x=warmup, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    
    # Marquer les moments d'adaptation
    for change_epoch in changes:
        ax2.axvline(x=change_epoch+1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Graphique sauvegardé: {output_path}")
    else:
        plt.show()
    
    # Statistiques
    print("\n" + "="*70)
    print("CLUSTER EVOLUTION STATISTICS")
    print("="*70)
    print(f"Initial clusters: {cluster_evolution[0]}")
    print(f"Final clusters: {cluster_evolution[-1]}")
    print(f"Net change: {cluster_evolution[-1] - cluster_evolution[0]:+d}")
    print(f"Number of adaptations: {len(changes)}")
    print(f"Min clusters: {min(cluster_evolution)}")
    print(f"Max clusters: {max(cluster_evolution)}")
    print("="*70)


if __name__ == "__main__":
    # Exemple d'utilisation
    import sys
    
    if len(sys.argv) > 1:
        train_info_path = sys.argv[1]
    else:
        # Par défaut, chercher dans results_warmup
        train_info_path = "results_warmup/mae_clustering_train_info.npy"
    
    if not os.path.exists(train_info_path):
        print(f"Erreur: fichier non trouvé: {train_info_path}")
        print("\nUsage: python plot_cluster_evolution.py [train_info_path]")
        print("Exemple: python plot_cluster_evolution.py results_warmup/mae_clustering_train_info.npy")
        sys.exit(1)
    
    # Définir le chemin de sortie
    output_dir = os.path.dirname(train_info_path)
    output_path = os.path.join(output_dir, 'cluster_evolution.png')
    
    # Générer le graphique
    plot_cluster_evolution(train_info_path, output_path)
