# Le but est d'inférer sur toutes les brdfs avec peu de données
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import csv

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import os.path as osp

from coordinateFunctions import *
from merlFunctions import *
import models
import util
from config import Config

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def extract_paths(folder, recursive=True):
    folder_path = Path(folder)

    if recursive:
        files = [p for p in folder_path.rglob("*") if p.is_file()]
    else:
        files = [p for p in folder_path.iterdir() if p.is_file()]

    # 1) Tous les paths (chemins complets)
    all_paths = [str(p.resolve()) for p in files]

    # 2) Noms sans extension ".binary" (uniquement si elle est présente)
    names_no_binary_ext = [p.stem for p in files]

    return all_paths, names_no_binary_ext


def compress_binary(mat, output, x_grid, NoL, NoV, encoder, size):
    z = util.encode_material_partial(mat, x_grid, NoL, NoV, encoder, size)
    np.save(output, z)
    return z


def decompress_binary(output, z, x_grid, NoL, NoV, decoder):
    brdf = util.decode_material(x_grid, NoL, NoV, z, decoder, False)
    util.save_BRDF(output, brdf)
    return brdf


def psnr(a: np.ndarray, b: np.ndarray, max_i: float) -> float:
    diff = a - b
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_i * max_i) / mse)


if __name__ == "__main__":
    size = 16200
    path_input = './brdfs/'
    path_latent_output = './latent_output/'
    path_binary_output = './binary_output/'

    x_grid = util.generate_coordinate()
    x_grid[:, :, 0:1] /= (np.pi)
    x_grid[:, :, 1:2] /= (np.pi/2)
    x_grid[:, :, 2:3] /= (np.pi/2)

    if Config.use_ringmap:
        x_grid = util.ringmap(x_grid)

    NoL = np.load('data/NdotL.npy')
    NoV = np.load('data/NdotV.npy')

    nps, encoder, decoder = models.get_nps()
    nps.load_weights('models/LOG4_Mean_Dim7/weight/Epoch40000_400000_weights_nps.h5')

    os.makedirs(osp.dirname(osp.abspath(path_binary_output)), exist_ok=True)
    os.makedirs(osp.dirname(osp.abspath(path_latent_output)), exist_ok=True)

    # Extract all binaries path
    all_paths, names = extract_paths(path_input)

    # Do csv to register new line in
    csv_path = "./output/resultats.csv"
    fieldnames = ["materiau", "PSNR"]
    file_exists = os.path.exists(csv_path)

    # Run on all binaries
    for i in tqdm(range(len(all_paths))):
        # compress
        output_latent = path_latent_output + names[i] + '_' + str(size) + '_latent_vector.npy'
        assert all_paths[i][-6:] == 'binary'
        brdf_ref = readMERLBRDF(all_paths[i])
        z = compress_binary(brdf_ref, output_latent, x_grid, NoL, NoV, encoder, size)

        # decompress
        output_binary = path_binary_output + names[i] + '_' + str(size) + '.binary'
        brdf_res = decompress_binary(output_binary, z, x_grid, NoL, NoV, decoder)

        # PSNR
        max_i = float(np.max(brdf_ref))
        overall = psnr(brdf_ref, brdf_res, max_i=max_i)

        # add to csv
        row_df = pd.DataFrame({'materiau': [names[i]], 'PSNR': [overall]})
        row_df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

