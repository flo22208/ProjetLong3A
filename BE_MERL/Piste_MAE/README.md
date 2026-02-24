# MAE pour MERL

Ce répertoire contient le code pour appliquer un auto-encodeur masqué sur les données de MERL. Ce code est une adaptation du celui contenu dans le répertoire "mae" fourni par Facebook (https://github.com/facebookresearch/mae). Il reprend aussi quelques codes du PFE de Gabriel Gournay.

## Description du code

Pour utiliser le code il faut placer le dataset de MERL dans merlDB/db/brdfs/. La base de donnée est accessible ici https://www.dropbox.com/scl/fo/ca477yl7gfibu0vg8ep99/AL1DfYSWogW2_Lokc1zFB1Y?rlkey=pot6hl4zyifdbwpnehr8ahm6p&e=1&dl=0 (attention, il y a un fichier readme dans le dossier "brdfs", à supprimer)

- ```mae_model_brdf.py``` (code adapté de celui de Facebook)

Contient le code du modèle MAE adapté aux données de MERL. Le modèle ne prends plus une image mais des données 3x90x90x180 en entrée. Le but du modèle est de masquer la valeur du triplet RBG pour certaines cases du cube. Le masquage est géré dans la fonction "random_masking", il est possible de le modifier.

NB: On peut préciser un poids en entrée qui correspond à la pondération de la loss dans le centre du tableau de valeur. Cette pondération peut-être adaptée dans l'init du modèle, à partir de la ligne 154.


- ```engine_pretrain.py``` (code de Facebook)

Fichier non-modifié, contient la fonction _train_one_epoch_ utile à l'entraînement.


- ```merlDB/database.py``` (code de Gabriel)

Fichier non-modifié, contient le code pour charger les données binaires des matériaux. Le plus important est la classe "DBuilder" ligne 131. Peut-être exécuté pour afficher un matériau au hasard.

- ```merlDB/utils.py``` (code de Gabriel)

Fichier non-modifié, contient le code de fonctions utilitaires pour le fichier ```merlDB/database.py```.

- ```merlDB/rendering.py``` (code de Gabriel)

Fichier non-modifié, contient le code de fonctions pour faire du rendu de matériaux.

- ```neural_data.py``` (code adapté à partir de celui de Gabriel)

Contient le code pour générer le dataset de MERL, classe "DBridge" ligne 31, applique une permutation aléatoire lors de la récupération d'un matériau (ligne 50, très utile sinon le modèle apprend le biais de la couleur dans l'espace latent). Contient la classe MAETrainer qui utilise la fonction d'entraînement du fichier ```engine_pretrain.py```.

- ```Notebook_entrainement_inference_rendu.ipynb``` 

Permet de faire l'entraînement sur toutes les données de MERL ainsi que de faire l'inférence et de générer le rendu pour un ou plusieurs matériaux.

- ```espace_latent.ipynb``` 

Permet de récupérer les repésentations latentes des matériaux à partir des poids du modèle et de les visualiser en 2D ou en 3D.