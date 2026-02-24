# Piste Sparse BRDF

Actuellement, ce répertoire contient le code pour inférer sur le modèle des représentations compactes avec un petit nombre d'observation. Ce code ne fonctionne pas seul et il faut remplacer ou ajouter les fichiers de ce répertoire au dossier github des représentations compactes que vous pouvez trouver ici (https://github.com/Rendering-at-ZJU/NPs-BRDF/tree/main).
Les chemins sont en durs dans les fichiers, il faut donc les changer avant de l'utiliser.

## Description du code

Pour utiliser le code il faut placer le dataset de MERL dans ./brdfs/

- ```inference_small.py```

Contient le code pour inférer toutes les BRDFs sur le modèle avec une certaine taille (fixe). Les résultats des espaces latents et des BRDFs reconstruites seront dans les dossiers 'latent_output' et 'binary_output' respectivement. Les PSNR entre les BRDF de référence et reconstruites sont calculés et placés dans le csv 'resultats.csv' dans le dossier output. Un exemple de ce csv est fourni.

- ```inference_var.py```

Contient le code pour inférer une BRDF avec différentes tailles sur le modèle. Les résultats des espaces latents et BRDF restituées sont sauvegarder dans les dossiers: 'var_latent_output' et 'var_binary_output'.Les PSNR entre les BRDF de référence et reconstruites sont calculés et placés dans le csv 'resultats_var.csv' dans le dossier output. Un exemple de ce csv est fourni.

- ```evaluate.py```

Ce fichier permet de faire le PSNR entre deux BRDFs.

- ```psnr_fig.py```

Ce fichier permet de faire la figure des PSNR en fonction des matériaux. Il se base sur le fichier './output/resultats.py'. Un exemple de cette figure se trouve dans le fichier, './output/psnr_plot.py'.

- ```psnr_var_fig.py``` (Facebook)

Ce fichier permet de faire la figure des PSNR en fonction de la taille des observations passée en entrée du modèle. Il se base sur le fichier './output/resultats_var.py'. Un exemple de cette figure se trouve dans le fichier, './output/psnr_var_plot.py'.

- ```util.py``` (adapté du code originel)

Ce fichier inclut une fonction en plus pour pouvoir faire l'inférence sur un certain nombre de données passé en paramètre. Cette fonction est utilisée par 'inference_small.py' et 'inference_var.py'. Il faut donc le remplacer dans le code originel. 

## Prochaines étapes

La nouvelle fonction ne permet que de prendre le début de la BRDF. Pour avoir des résultats plus représentatifs, il peut être pertinent de la modifier.




