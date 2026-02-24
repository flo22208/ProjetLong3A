"""
Script pour tester et comparer les rendus avec/sans log-remapping
"""

import merlFunctions as mf
import merlFunctions_nino as mf_nino
import numpy as np
import os

MATERIAL_FILE = "./merlDB/db/brdfs/blue-acrylic.binary"

print("="*80)
print("TEST: Comparaison log-remapping vs linear normalization")
print("="*80)

# Charger le BRDF
brdf = mf.readMERLBRDF(MATERIAL_FILE)
print(f"\nBRDF loaded, shape: {brdf.shape}")

# Transposer comme dans loadScene.py
brdf = np.transpose(brdf, (1, 2, 0, 3))  # merl format
print(f"Transposed shape: {brdf.shape}")

scene_dir = "./matpreview"
if not os.path.exists(scene_dir):
    os.makedirs(scene_dir)

# Test 1: Sans log-remapping (nouvelle version par défaut)
print("\n" + "="*80)
print("TEST 1: LINEAR NORMALIZATION (nouvelle version)")
print("="*80)
temp_file_linear = os.path.join(scene_dir, "temp_brdf_linear.bsdf")
mf_nino.saveBSDF(temp_file_linear, brdf, debug=False, use_log_remapping=False)

# Test 2: Avec log-remapping (ancienne version)
print("\n" + "="*80)
print("TEST 2: LOG-SPACE REMAPPING (ancienne version)")
print("="*80)
temp_file_log = os.path.join(scene_dir, "temp_brdf_log.bsdf")
mf_nino.saveBSDF(temp_file_log, brdf, debug=False, use_log_remapping=True)

print("\n" + "="*80)
print("RÉSULTATS")
print("="*80)
print(f"\nFichiers générés:")
print(f"  1. {temp_file_linear} (normalisation linéaire)")
print(f"  2. {temp_file_log} (remapping log-space)")
print(f"\nPour tester le rendu, modifier loadScene.py pour utiliser:")
print(f"  - temp_brdf_linear.bsdf (recommandé)")
print(f"  - temp_brdf_log.bsdf (ancienne méthode)")
print(f"\nLa version linéaire devrait éliminer les contours blancs saturés!")
