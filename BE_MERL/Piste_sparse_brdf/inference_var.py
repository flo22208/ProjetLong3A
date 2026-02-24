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
    size = [10*16200, 3*16200, 16200, int(16200/2), int(16200/5)]
    name = "alum_bronze"
    path_input = './brdfs/alum-bronze.binary'
    path_latent_output = './var_latent_output/'
    path_binary_output = './var_binary_output/'

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

    # Do csv to register new line in
    csv_path = "./output/resultats_var.csv"
    fieldnames = ["materiau", "size", "PSNR"]
    file_exists = os.path.exists(csv_path)

    # Run on all binaries
    for i in tqdm(size):
        # compress
        output_latent = path_latent_output + '_' + name + '_' + str(i) + '_latent_vector.npy'
        brdf_ref = readMERLBRDF(path_input)
        z = compress_binary(brdf_ref, output_latent, x_grid, NoL, NoV, encoder, i)

        # decompress
        output_binary = path_binary_output + '_' + name + '_' + str(i) + '.binary'
        brdf_res = decompress_binary(output_binary, z, x_grid, NoL, NoV, decoder)

        # PSNR
        max_i = float(np.max(brdf_ref))
        overall = psnr(brdf_ref, brdf_res, max_i=max_i)

        # add to csv
        row_df = pd.DataFrame({'materiau': ["alum-bronze"], 'size': [i], 'PSNR': [overall]})
        row_df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

