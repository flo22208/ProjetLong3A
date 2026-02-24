"""
Module de chargement des données MERL pour l'entraînement
Ce module fournit des fonctions pour charger et préparer les datasets MERL
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
import merlDB.database as db
import neural_data as nd
from torch.utils.data import Dataset


def rgb_to_lab(rgb_tensor):
    """
    Convertit un tensor RGB en LAB
    
    Args:
        rgb_tensor: Tensor de forme (B, 3, D, H, W) ou (3, D, H, W) avec valeurs dans [0, 1]
    
    Returns:
        Tensor LAB de même forme avec:
        - L dans [0, 100]
        - A dans [-128, 127]
        - B dans [-128, 127]
    """
    # Conversion RGB → XYZ
    # Matrice de conversion sRGB → XYZ (illuminant D65)
    rgb = rgb_tensor.clone()
    
    # Linearisation sRGB (gamma correction inverse)
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92
    )
    
    # Matrice de transformation RGB → XYZ
    # Pour sRGB avec illuminant D65
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # Reshape pour multiplication matricielle
    original_shape = rgb_linear.shape
    if len(original_shape) == 5:  # (B, 3, D, H, W)
        B, C, D, H, W = original_shape
        rgb_flat = rgb_linear.reshape(B, 3, -1)  # (B, 3, D*H*W)
        xyz_flat = torch.matmul(M, rgb_flat)  # (B, 3, D*H*W)
        xyz = xyz_flat.reshape(B, 3, D, H, W)
    else:  # (3, D, H, W)
        C, D, H, W = original_shape
        rgb_flat = rgb_linear.reshape(3, -1)  # (3, D*H*W)
        xyz_flat = torch.matmul(M, rgb_flat)  # (3, D*H*W)
        xyz = xyz_flat.reshape(3, D, H, W)
    
    # Normalisation par le point blanc D65
    # Pour D65: Xn=0.95047, Yn=1.00000, Zn=1.08883
    xyz_normalized = xyz.clone()
    xyz_normalized[..., 0, :, :, :] = xyz[..., 0, :, :, :] / 0.95047  # X
    xyz_normalized[..., 1, :, :, :] = xyz[..., 1, :, :, :] / 1.00000  # Y
    xyz_normalized[..., 2, :, :, :] = xyz[..., 2, :, :, :] / 1.08883  # Z
    
    # Fonction f(t) pour la conversion XYZ → LAB
    delta = 6.0 / 29.0
    delta_cube = delta ** 3
    
    def f(t):
        mask = t > delta_cube
        return torch.where(
            mask,
            torch.pow(t, 1.0/3.0),
            t / (3 * delta**2) + 4.0/29.0
        )
    
    fx = f(xyz_normalized[..., 0, :, :, :])
    fy = f(xyz_normalized[..., 1, :, :, :])
    fz = f(xyz_normalized[..., 2, :, :, :])
    
    # Calcul des composantes LAB
    L = 116.0 * fy - 16.0
    A = 500.0 * (fx - fy)
    B = 200.0 * (fy - fz)
    
    # Reconstruction du tensor LAB
    lab = torch.stack([L, A, B], dim=-4)
    
    return lab


def lab_to_rgb(lab_tensor):
    """
    Convertit un tensor LAB en RGB (fonction inverse pour vérification)
    
    Args:
        lab_tensor: Tensor de forme (B, 3, D, H, W) ou (3, D, H, W)
    
    Returns:
        Tensor RGB de même forme avec valeurs dans [0, 1]
    """
    L = lab_tensor[..., 0, :, :, :]
    A = lab_tensor[..., 1, :, :, :]
    B = lab_tensor[..., 2, :, :, :]
    
    # LAB → XYZ
    fy = (L + 16.0) / 116.0
    fx = A / 500.0 + fy
    fz = fy - B / 200.0
    
    delta = 6.0 / 29.0
    
    def f_inv(t):
        mask = t > delta
        return torch.where(
            mask,
            torch.pow(t, 3.0),
            3 * delta**2 * (t - 4.0/29.0)
        )
    
    xyz_normalized = torch.stack([f_inv(fx), f_inv(fy), f_inv(fz)], dim=-4)
    
    # Dénormalisation par le point blanc D65
    xyz = xyz_normalized.clone()
    xyz[..., 0, :, :, :] = xyz_normalized[..., 0, :, :, :] * 0.95047
    xyz[..., 1, :, :, :] = xyz_normalized[..., 1, :, :, :] * 1.00000
    xyz[..., 2, :, :, :] = xyz_normalized[..., 2, :, :, :] * 1.08883
    
    # XYZ → RGB linéaire
    M_inv = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], device=lab_tensor.device, dtype=lab_tensor.dtype)
    
    original_shape = xyz.shape
    if len(original_shape) == 5:  # (B, 3, D, H, W)
        B, C, D, H, W = original_shape
        xyz_flat = xyz.reshape(B, 3, -1)
        rgb_linear_flat = torch.matmul(M_inv, xyz_flat)
        rgb_linear = rgb_linear_flat.reshape(B, 3, D, H, W)
    else:  # (3, D, H, W)
        C, D, H, W = original_shape
        xyz_flat = xyz.reshape(3, -1)
        rgb_linear_flat = torch.matmul(M_inv, xyz_flat)
        rgb_linear = rgb_linear_flat.reshape(3, D, H, W)
    
    # Gamma correction (linéaire → sRGB)
    mask = rgb_linear > 0.0031308
    rgb = torch.where(
        mask,
        1.055 * torch.pow(rgb_linear, 1.0/2.4) - 0.055,
        12.92 * rgb_linear
    )
    
    # Clamp pour s'assurer que les valeurs sont dans [0, 1]
    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    return rgb


class LuminanceOnlyDataset(Dataset):
    """
    Wrapper de dataset qui convertit RGB en LAB et ne retourne que le canal de luminance (L)
    """
    def __init__(self, base_dataset, normalize_luminance=True):
        """
        Args:
            base_dataset: Dataset de base retournant des données RGB
            normalize_luminance: Si True, normalise L de [0, 100] vers [0, 1]
        """
        self.base_dataset = base_dataset
        self.normalize_luminance = normalize_luminance
    
    def __len__(self):
        return len(self.base_dataset) if hasattr(self.base_dataset, '__len__') else 0
    
    def __getitem__(self, idx):
        # Récupérer l'item du dataset de base
        item = self.base_dataset[idx]
        
        # Gérer différents formats de sortie du dataset
        if isinstance(item, dict):
            # Format dictionnaire (ex: {'values': tensor})
            rgb_data = item['values']
            is_dict = True
        elif isinstance(item, (tuple, list)):
            # Format tuple/liste (data, label)
            rgb_data = item[0]
            other_data = item[1:]
            is_dict = False
        else:
            # Format tensor direct
            rgb_data = item
            other_data = None
            is_dict = False
        
        # Convertir RGB en LAB
        lab_data = rgb_to_lab(rgb_data)
        
        # Extraire uniquement le canal L (luminance)
        # Shape: (B, 3, D, H, W) → (B, 1, D, H, W) ou (3, D, H, W) → (1, D, H, W)
        luminance = lab_data[..., 0:1, :, :, :]
        
        # Normaliser si demandé
        if self.normalize_luminance:
            luminance = luminance / 100.0  # [0, 100] → [0, 1]
        
        # Retourner dans le même format que l'entrée
        if is_dict:
            result = item.copy()
            result['values'] = luminance
            return result
        elif 'other_data' in locals() and other_data is not None:
            return (luminance,) + other_data
        else:
            return luminance


class MERLDataConfig:
    """Configuration pour le chargement des données MERL"""
    def __init__(self):
        # Si c'est vide ça veut dire qu'on fait l'entraînement sur tous les matériaux
        self.materiaux = ""
        
        # Pourcentage de triplets (R,G,B) masqués (recommandé 0.5)
        self.ratio_masking = 0.5
        
        # Quantité de données à utiliser pour l'entraînement (min 1 max 100)
        self.train_size = 100
        
        # Batch size
        self.batch_size = 20
        
        # Chemins par défaut
        self.merldir = 'merlDB/db/brdfs/'
        self.mediandir = 'merlDB/db/'
        self.outdir = 'results/'
        
        # Options de chargement
        self.monomat = None
        self.load_previous = False


def get_data_args_parser():
    """Parser d'arguments pour le chargement des données"""
    parser = argparse.ArgumentParser('MERL Data Loader', add_help=False)
    
    # Chemins des données
    parser.add_argument('--outdir', default='results/', type=str,
                        help='Chemin de sortie pour sauvegarder les infos')
    parser.add_argument('--merldir', default='merlDB/db/brdfs/', type=str,
                        help='Chemin vers la base de données MERL')
    parser.add_argument('--mediandir', default='merlDB/db/', type=str,
                        help='Chemin vers le fichier median MERL')
    
    # Paramètres de données
    parser.add_argument('--train_size', default=100, type=int,
                        help='Nombre de matériaux pour l\'entraînement')
    parser.add_argument('--monomat', default=None, type=str,
                        help='Si spécifié, charge uniquement ce matériau')
    parser.add_argument('--materiau', default="", type=str,
                        help='Matériau spécifique à charger')
    
    # Options de chargement
    parser.add_argument('--load_previous', action='store_true',
                        help='Charger les matériaux d\'un entraînement précédent')
    parser.add_argument('--seed', default=0, type=int,
                        help='Graine aléatoire pour la sélection des matériaux')
    parser.add_argument('--luminance_only', action='store_true',
                        help='Convertir en LAB et utiliser uniquement le canal de luminance')
    
    return parser


def median_mapping_factory(medians, epsilon=0.002):
    """
    Crée une fonction de mapping pour les albedos
    
    Args:
        medians: Valeurs médianes MERL
        epsilon: Valeur epsilon pour éviter log(0)
    
    Returns:
        Fonction de mapping
    """
    def median_mapping(albedos):
        return np.log((albedos + epsilon)/(medians + epsilon) + 1)
    return median_mapping


def load_merl_datasets(args=None, device=None, classes_dict=None):
    """
    Charge les datasets MERL pour l'entraînement et le test
    
    Args:
        args: Arguments parsés (si None, utilise les valeurs par défaut)
        device: Device PyTorch (si None, détecte automatiquement)
        classes_dict: Dictionnaire de classes et de matériaux (optionnel)
    
    Returns:
        dict contenant:
            - 'train_dataset': Dataset d'entraînement
            - 'test_dataset': Dataset de test
            - 'train_mats': Liste des matériaux d'entraînement
            - 'test_mats': Liste des matériaux de test
            - 'device': Device utilisé
            - 'medians': Valeurs médianes MERL
    """
    
    # Détection du device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parser les arguments si non fournis
    if args is None:
        parser = get_data_args_parser()
        args, _ = parser.parse_known_args()
    
    # Chargement des médianes
    print(f"Loading medians from {args.mediandir}")
    medians = db.readbin(args.mediandir + 'merl_median.binary')
    median_mapping = median_mapping_factory(medians)
    
    # Création des builders
    print("Creating data builders...")
    dbuilder = db.DBuilder(db_path=args.merldir, albedo_mapping=median_mapping)
    test_dbuilder = db.DBuilder(db_path=args.merldir, albedo_mapping=median_mapping)
    
    # Liste des matériaux disponibles
    ldb = dbuilder.list_db()
    print(f"Found {len(ldb)} materials in database")
    
    # Sélection des matériaux
    if args.load_previous:
        print('Loading materials from previous training...')
        train_info = np.load(args.outdir + 'train_info.npy', allow_pickle=True).item()
        train_mats = train_info.get('train_mats')
        test_mats = train_info.get('test_mats', [])
    else:
        if args.materiau != "":
            # Un seul matériau spécifié
            train_mats = [args.materiau]
            test_mats = [args.materiau]
        elif args.monomat is not None:
            # Mode mono-matériau
            train_mats = [args.monomat]
            test_mats = []
        else:
            if classes_dict is not None:
                # classes_dict est une liste de noms de classes
                # Les matériaux contenant le nom de la classe sont automatiquement assignés
                np.random.seed(args.seed)
                train_mats = []
                test_mats = []
                train_ratio = 0.9
                
                if isinstance(classes_dict, list) and len(classes_dict) > 0:
                    # Pour chaque classe, trouver les matériaux contenant le nom de la classe ### les matériaux other sont supprimés
                    for class_name in classes_dict:
                        # Filtrer les matériaux qui contiennent le nom de la classe (case-insensitive)
                        class_materials = [m for m in ldb if class_name.lower() in m.lower()]
                        
                        if not class_materials:
                            print(f"Warning: No materials found for class '{class_name}'")
                            continue
                        
                        # Split train/test pour cette classe
                        shuffled = np.random.permutation(class_materials).tolist()
                        train_count = max(1, int(len(shuffled) * train_ratio))
                        train_split = shuffled[:train_count]
                        test_split = shuffled[train_count:]
                        
                        train_mats.extend(train_split)
                        test_mats.extend(test_split)
                        
                        print(f"Class '{class_name}': {len(class_materials)} materials ({len(train_split)} train, {len(test_split)} test)")
            else:
                # Sélection aléatoire des matériaux
                np.random.seed(args.seed)
                train_size = min(int(args.train_size), len(ldb))
                mats_idx = np.random.choice(range(len(ldb)), train_size, replace=False)
                train_mats = [ldb[idx] for idx in mats_idx]
                test_mats = [mat for mat in ldb if mat not in train_mats]

    
    print(f"\nTraining materials ({len(train_mats)}): {train_mats[:5]}..." if len(train_mats) > 5 else f"\nTraining materials: {train_mats}")
    print(f"Test materials ({len(test_mats)}): {test_mats[:5]}..." if len(test_mats) > 5 else f"Test materials: {test_mats}")
    
    # Chargement des matériaux d'entraînement
    print("\nLoading training materials...")
    for mat in tqdm(train_mats, desc='Loading train mats'):
        if mat != "Rea":
            dbuilder.load_mat(mat)
    
    # Chargement des matériaux de test
    test_dataset = None
    if torch.cuda.is_available() and len(test_mats) > 0:
        print("\nLoading test materials...")
        for mat in tqdm(test_mats, desc='Loading test mats'):
            if mat != "Rea":
                test_dbuilder.load_mat(mat)
        test_dataset = nd.DBridge(test_dbuilder, device)
    
    # Création des datasets
    print("\nCreating datasets...")
    train_dataset = nd.DBridge(dbuilder, device)
    
    # Application du wrapper luminance si demandé
    if hasattr(args, 'luminance_only') and args.luminance_only:
        print("Applying luminance-only transformation (RGB → LAB → L channel)")
        train_dataset = LuminanceOnlyDataset(train_dataset, normalize_luminance=True)
        if test_dataset is not None:
            test_dataset = LuminanceOnlyDataset(test_dataset, normalize_luminance=True)
    
    # Nettoyage
    del dbuilder
    del test_dbuilder
    
    print(f"\nDatasets created successfully!")
    print(f"Train dataset size: {len(train_dataset) if hasattr(train_dataset, '__len__') else 'N/A'}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset) if hasattr(test_dataset, '__len__') else 'N/A'}")
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_mats': train_mats,
        'test_mats': test_mats,
        'device': device,
        'medians': medians
    }


def save_dataset_info(data_dict, output_path='results/data_info.npy'):
    """
    Sauvegarde les informations sur les datasets chargés
    
    Args:
        data_dict: Dictionnaire retourné par load_merl_datasets
        output_path: Chemin pour sauvegarder les informations
    """
    info = {
        'train_mats': data_dict['train_mats'],
        'test_mats': data_dict['test_mats'],
        'device': str(data_dict['device'])
    }
    np.save(output_path, info)
    print(f"Dataset info saved to {output_path}")


def load_luminance_datasets(args=None, device=None, classes_dict=None):
    """
    Charge les datasets MERL en mode luminance uniquement (canal L de LAB)
    
    Args:
        args: Arguments parsés (si None, utilise les valeurs par défaut)
        device: Device PyTorch (si None, détecte automatiquement)
        classes_dict: Dictionnaire de classes et de matériaux (optionnel)
    
    Returns:
        dict contenant les datasets avec uniquement le canal de luminance
    """
    # Créer ou modifier args pour activer luminance_only
    if args is None:
        parser = get_data_args_parser()
        args, _ = parser.parse_known_args()
    
    # Forcer le mode luminance
    args.luminance_only = True
    
    # Charger les datasets normalement
    return load_merl_datasets(args=args, device=device, classes_dict=classes_dict)


if __name__ == "__main__":
    # Exemple d'utilisation du module
    print("="*50)
    print("MERL Data Loader - Example Usage")
    print("="*50)
    
    # Test 1: Charger les datasets RGB normaux
    print("\n### Test 1: Loading RGB datasets ###")
    data = load_merl_datasets()
    
    # Sauvegarder les informations
    save_dataset_info(data)
    
    # Test 2: Charger les datasets en mode luminance uniquement
    print("\n" + "="*50)
    print("### Test 2: Loading Luminance-only datasets ###")
    print("="*50)
    
    parser = get_data_args_parser()
    args, _ = parser.parse_known_args(['--luminance_only', '--train_size', '5'])
    data_lum = load_merl_datasets(args=args)
    
    print("\n### Test 3: Testing RGB to LAB conversion ###")
    # Créer un tensor RGB de test
    test_rgb = torch.rand(1, 3, 10, 10, 10)  # Batch de test
    print(f"Test RGB shape: {test_rgb.shape}")
    print(f"RGB range: [{test_rgb.min():.3f}, {test_rgb.max():.3f}]")
    
    # Conversion RGB → LAB
    test_lab = rgb_to_lab(test_rgb)
    print(f"LAB shape: {test_lab.shape}")
    print(f"L range: [{test_lab[:, 0].min():.3f}, {test_lab[:, 0].max():.3f}]")
    print(f"A range: [{test_lab[:, 1].min():.3f}, {test_lab[:, 1].max():.3f}]")
    print(f"B range: [{test_lab[:, 2].min():.3f}, {test_lab[:, 2].max():.3f}]")
    
    # Conversion LAB → RGB (vérification)
    test_rgb_back = lab_to_rgb(test_lab)
    print(f"RGB reconstructed shape: {test_rgb_back.shape}")
    print(f"RGB reconstructed range: [{test_rgb_back.min():.3f}, {test_rgb_back.max():.3f}]")
    print(f"Reconstruction error (MSE): {torch.mean((test_rgb - test_rgb_back)**2):.6f}")
    
    print("\nData loading complete!")
