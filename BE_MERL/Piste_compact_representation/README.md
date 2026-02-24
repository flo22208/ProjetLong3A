# Piste représentations latentes de BRDFs mesurées à l’aide de processus neuronaux

Pour lancer l'entraînement du svm : 
```
python3 ./train_classifier.py --latent-dir ./latent_vectors/ --output-dir ./models
```
Avec --latent-dir : le dossier contenant les représentations latentes 7D

Et --output-dir : le dossier de sortie pour le modèle

Pour lancer l'inférence : 
```
python3 inference.py  --classifier-path ./models/svm_classifier.pkl --scaler-path ./models/scaler.pkl yellow-plastic
```
Avec --classifier_path : le chemin vers le classifieur

Et --output-dir : le chemin vers le scaler --scaler-path

Et en dernier argument le nom du matériau
