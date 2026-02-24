
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Définir les catégories

CATEGORIES = {
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
    ],
    
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
    
    'metallic_paint': [
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
    
    'two_layer': [
        'two-layer-gold',
        'two-layer-silver'
    ],
    
    'phenolic': [
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
        'pearl-paint'
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
        'teflon'
    ],
    
    'rubber': [
        'blue-rubber',
        'neoprene-rubber',
        'pure-rubber',
        'violet-rubber',
        'green-latex'
    ],
    
    'acrylic': [
        'blue-acrylic',
        'green-acrylic',
        'violet-acrylic',
        'white-acrylic'
    ],
    
    'wood': [
        'cherry-235',
        'colonial-maple-223',
        'fruitwood-241',
        'ipswich-pine-221',
        'natural-209',
        'pickled-oak-260',
        'special-walnut-224'
    ],
    
    'stone_mineral': [
        'aventurnine',
        'black-obsidian',
        'hematite',
        'pink-jasper',
        'white-marble'
    ],
    
    'special': [
        'silicon-nitrade',
        'white-diffuse-bball'
    ]
}


def load_latent_vectors(latent_dir):
    """Charger tous les vecteurs latents"""
    print(f"Chargement des vecteurs latents depuis: {latent_dir}")
    
    X_latent = []
    y_labels = []
    filenames = []
    
    # Mapping material -> category
    material_to_category = {}
    for category, materials in CATEGORIES.items():
        for material in materials:
            material_to_category[material] = category
    
    latent_files = list(Path(latent_dir).glob('*_latentVector.npy'))
    print(f"  Trouvé {len(latent_files)} fichiers")
    
    found_categories = set()
    not_found = []
    load_errors = []
    
    for filepath in sorted(latent_files):
        material_name = filepath.stem.replace('_latentVector', '')
        category = material_to_category.get(material_name)
        
        if category is None:
            not_found.append(material_name)
            continue
        
        latent_array = np.load(filepath)
        latent_vector = latent_array[0].squeeze()                  
        X_latent.append(latent_vector)
        y_labels.append(category)
        filenames.append(material_name)
        found_categories.add(category)
    
    print(f"\nChargement terminé:")

    X_latent_array = np.array(X_latent)
    
    return X_latent_array, np.array(y_labels), filenames

def train_classifier(latent_dir, output_dir='./models'):
    """Entraîner le classifieur"""

    # Charger les vecteurs
    X_latent, y_labels, filenames = load_latent_vectors(latent_dir)
    
    if len(X_latent) == 0:
        print("Aucun vecteur latent chargé!")
        return
    
    print(f"Shape: {X_latent.shape}")
    print(f"Classes: {np.unique(y_labels)}")

    
    unique, counts = np.unique(y_labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {cls:20s}: {cnt:3d} échantillons")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Encoder les labels en nombres**
    print(f"Encodage des labels...")
    label_encoder = LabelEncoder()
    y_labels_encoded = label_encoder.fit_transform(y_labels)
    
    print(f"Labels encodés:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"    {i} -> {cls}")
    
    # Sauvegarder le label encoder
    label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder sauvegardé: {label_encoder_path}")
    
    # Diviser en train/test
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
        X_latent, y_labels, y_labels_encoded,
        test_size=0.2, 
        stratify=y_labels, 
        random_state=42
    )

    
    print(f"Train: {len(X_train)} échantillons")
    print(f"Test: {len(X_test)} échantillons")
    
    # Normaliser
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler sauvegardé: {scaler_path}")
    
    results = {}
    
    print("Entraînement SVM")

    svm_clf = SVC(
        kernel='rbf', 
        C=10.0, 
        gamma='scale', 
        probability=True, 
        random_state=42
    )

    print("  Entraînement en cours...")
    svm_clf.fit(X_train_scaled, y_train)  
    svm_pred = svm_clf.predict(X_test_scaled)
    
    print("Résultats SVM:")
    print(classification_report(y_test, svm_pred, zero_division=0))
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"Accuracy: {svm_acc:.4f}")
    
    svm_path = os.path.join(output_dir, 'svm_classifier.pkl')
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_clf, f)
    print(f"\nSVM sauvegardé: {svm_path}")
    
    results = (svm_clf, svm_pred, svm_acc)
    
    print(f"\nFichiers sauvegardés dans: {output_dir}/")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--latent-dir',
        type=str,
        default='./latent_vectors',
        help='Dossier contenant les fichiers *_latentVector.npy'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Dossier de sortie pour les modèles'
    )
    
    args = parser.parse_args()
    
    train_classifier(args.latent_dir, args.output_dir)