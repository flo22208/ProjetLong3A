import torch
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import merlDB.database as db
from tqdm import tqdm
import random
from torchsummary import torchsummary
from mae_model_brdf import MaskedAutoencoderViT3D
from engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler

INPUT_SIZE = 3*db.RES_PHI_D*db.RES_THETA_D*db.RES_THETA_H
INPUT_SHAPE = (3,db.RES_THETA_D, db.RES_THETA_H, db.RES_PHI_D)


class Samples(torchdata.Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def add_samples(self, brdf_array):
        self.inputs.append(brdf_array)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index]
    
class DBridge(torchdata.Dataset):
    def __init__(self, dbuilder : db.DBuilder, device, mask_value=1):
        super().__init__()
        self.samples = Samples()
        self.mats = []
        self.labels =[]
        self.sigmoid = nn.Sigmoid()
        self.mask_value = mask_value
        for mat in dbuilder.list_db_loaded():
            array = torch.from_numpy(dbuilder.brdf_data(mat)).float().to(device)
            array = torch.swapaxes(array, 0, -1)
            self.samples.add_samples(array)
            self.mats.append(mat)
            self.labels.append(db.find_label(db.LABELS_MAT, mat))
    def __len__(self):
        return self.samples.__len__()
    
    def __getitem__(self, index):
        values = self.samples.__getitem__(index)
        values = values[torch.randperm(3),:,:,:]
        values = torch.maximum(torch.tensor(0), values)
        if self.mask_value < 1:
            n = values.numel()
            nb_mask = int(self.mask_value * n)
            mask_ind = torch.randperm(n)[:nb_mask]
            masked_values = values.clone()
            masked_values.reshape(-1)[mask_ind] = -1
        else:
            masked_values = values
        return {'mat': self.mats[index], 'label':self.labels[index], 'values':masked_values}
    

def custom_collate_fn(batch):
    mats = [item['mat'] for item in batch]                     
    labels = torch.tensor([item['label'] for item in batch])   
    values = torch.stack([item['values'] for item in batch]) 
    return {'mat': mats, 'label': labels, 'values': values}

class MAETrainer():
    def __init__(self, device, dataset: torchdata.Dataset, mae=None, batch_size=1, shuffle=False, lr=1e-4, args=None):
        self.args = args
        self.dataset = dataset
        if mae is None:
            self.mae = MaskedAutoencoderViT3D(batch_size=batch_size)
        else:
            self.mae = mae

        self.device = device
        self.batch_size = batch_size
        self.mae.to(device)

        self.loss_scaler = NativeScaler()
        self.epoch = 0
    

        self.decoder_loss = nn.MSELoss()

        self.optim = torch.optim.Adam(mae.parameters(), lr=lr)
        self.dataloader = torchdata.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
        self.losses = []
    
    def train_epoch(self):
        self.mae.train()
        total_ae_loss = 0

        train_stats, avg_loss = train_one_epoch(self.mae, self.dataloader, self.optim, self.device, self.epoch, self.loss_scaler, log_writer=None, args=self.args)
        
        total_ae_loss /= len(self.dataloader.dataset)
        self.losses.append(avg_loss)
        print('loss', avg_loss)
        self.epoch += 1
        return avg_loss
        
