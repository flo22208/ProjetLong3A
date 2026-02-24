import numpy as np
import pickle
import os
from pathlib import Path

class SimpleBRDFClassifier:
    """Classifieur utilisant les vecteurs latents pré-calculés"""
    
    def __init__(self, latent_dir, classifier_path, scaler_path):
        self.latent_dir = latent_dir
        
        # Charger le classifieur
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Charger le scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Classifieur chargé")
        print(f"  Classes: {self.classifier.classes_}")
    
    def load_latent_vector(self, material_name):
        """Charger un vecteur latent pré-calculé"""
        filepath = os.path.join(self.latent_dir, f"{material_name}_latentVector.npy")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vecteur latent non trouvé: {filepath}")
        
        latent = np.load(filepath)
        
        z_mean = latent[0].squeeze()
 
        return z_mean
    
    def predict(self, material_name):
        """Prédire la classe d'un matériau"""
        print(f"\nChargement du vecteur latent: {material_name}")
        
        # Charger le vecteur
        z_mean = self.load_latent_vector(material_name)
        print(f"  Vecteur: {z_mean}")
        
        # Normaliser
        z_scaled = self.scaler.transform(z_mean.reshape(1, -1))
        
        # Prédire
        prediction = self.classifier.predict(z_scaled)[0]
        
        result = {
            'prediction': prediction,
            'latent_vector': z_mean
        }
        
        return result

# Utilisation
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('material_name', help='Nom du matériau (ex: alum-bronze)')
    parser.add_argument('--latent-dir', default='./latent_vectors')
    parser.add_argument('--classifier-path', default='./models/svm_classifier.pkl')
    parser.add_argument('--scaler-path', default='./models/scaler.pkl')
    
    args = parser.parse_args()
    
    classifier = SimpleBRDFClassifier(
        args.latent_dir,
        args.classifier_path,
        args.scaler_path
    )
    
    result = classifier.predict(args.material_name)
    
    print(f"Prédiction: {result['prediction']}")