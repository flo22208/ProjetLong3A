import torch
import torch.nn as nn
import math
import tqdm
import requests
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from utilsTrain import create_rusinkiewicz_grid, disney_brdf_torch

# Parameters
EPOCHS = 5
BATCH_SIZE = 64
PATH_MODEL = f"./model_encoder_disney.pt"


# print statistics 
print("Number of epochs :", EPOCHS)
print("Batch size:", BATCH_SIZE)

# --- DEFINE THE MODEL ---

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

model.eval().to(device)

# --- TRAINING ---

print(f"--- Training on {device} ---")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

param_dim = 13

wi, wo, N = create_rusinkiewicz_grid(device)

for epoch in range(EPOCHS):

    # ---- Sample random parameters in [0,1]
    params = torch.rand(BATCH_SIZE, param_dim, device=device)

    # ---- Generate analytic ground truth on the fly
    with torch.no_grad():
        gt = disney_brdf_torch(params, wi, wo, N)

    # ---- Forward pass
    pred = model(params)

    # ---- Compute loss
    loss = criterion(pred, gt)

    # ---- Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")