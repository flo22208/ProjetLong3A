import mitsuba as mi
import os
import merlFunctions as mf
import numpy as np
from mitsuba_renderer import convert_merl_to_bsdf
import merlFunctions_nino as mf_nino

#ATERIAL_FILE = "./results/prediction/red-plastic-pred.binary"
MATERIAL_FILE = "./merlDB/db/brdfs/blue-acrylic.binary"
SPP = 2048

pred = MATERIAL_FILE.__contains__("pred")
brdf = mf.readMERLBRDF(MATERIAL_FILE)
#print("BRDF loaded, shape:", brdf.shape)
scene_dir = "./matpreview"
if not os.path.exists(scene_dir):
    os.makedirs(scene_dir)
temp_file = os.path.join(scene_dir, "temp_brdf.bsdf")
# save au format BSDF de Mitsuba
# shape -> pred : (90, 180, 90, 3), merl : (180, 90, 90, 3)
if pred:
    brdf = np.transpose(brdf, (2, 0, 1, 3)) # pred
else:
    brdf = np.transpose(brdf, (1, 2, 0, 3)) # merl
#mf.saveBSDF(temp_file, brdf) # fonction de sauvegarde au format BSDF de Mitsuba
mf_nino.saveBSDF(temp_file, brdf) # Linear normalization (default, avoids saturation)
#convert_merl_to_bsdf(MATERIAL_FILE, temp_file) # fonction de conversion au format BSDF de Mitsuba


### Mitsuba loading and rendering code
mi.set_variant('cuda_ad_rgb')

SCENE = 'matpreview/scene_2.xml'

scene = mi.load_file(SCENE)

image = mi.render(scene, spp=SPP)

# Sauvegarder l'image rendue
mi.util.write_bitmap('blue_acrylic_nino_anisotrop.png', image)