
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ChainedScheduler, SequentialLR
from sklearn.cluster import KMeans

from utils.dataset import CLEVERDataset
from utils.models import Encoder, Decoder
from utils.models_unet import ResBlock, SpatialTransformer, UNET, SinusoidalPositionEmbeddings
from utils.utilities import create_folder, store_json_file, plot_loss_graph, store_reconstructed_mask_image
from utils.diffusion_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import random
import os

from vae_ import VAE 

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is ", device)

# TO EDIT ZONE --------------------------------------------
#####Hyperparameters
BATCH_SIZE = 1
NUM_WORKERS = 16

folder_dir = "/home/tkarthikeyan/IIT_DELHI/COL775_Deep_Learning/Assignment_2/dataset"

val_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "val")
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 16)

with torch.no_grad():
    #Loading VAE checkpoint
    vae = VAE().to(device)
    ckpt = torch.load("vae_checkpoint.pth")
    vae.load_state_dict(ckpt)

    for param in vae.parameters():
        param.requires_grad = False

    #DIFFUSION PARAMS
    betas = linear_beta_schedule()

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    SP = SinusoidalPositionEmbeddings(512)

    # criterion = nn.MSELoss()

    encoder = Encoder().to(device)
    decoder = UNET().to(device)

    model = torch.load("/home/tkarthikeyan/IIT_DELHI/COL775_Deep_Learning/2023AIZ8140_2023AIZ8073/part2_model.pth")
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"])
    print("slot diffusion model loaded!")

    k = 3
    for data in tqdm(val_data_loader):
        image = data.to(device)
        
        image = image.to(device)

        slots, attn_masks = encoder(image)
        attn_masks = attn_masks.reshape((1,11,32,32))
        rescaled_attn_masks = F.interpolate(attn_masks, size=(128,128), mode='bilinear', align_corners=False)

        temp_out = torch.randn((1, 3, 32, 32)).to(device)

        img_lst = []
        index_lst = [0, 50, 100, 200, 350, 500, 600, 700, 800, 900, 950, 999]
        for ts in tqdm(reversed(range(0, 1000))):
            sinusoidal_timestep = SP(torch.tensor([ts])).to(device)
            unet_out = decoder(temp_out, sinusoidal_timestep, slots)
            temp_out = p_sample(unet_out, temp_out, torch.full((1,), ts, device=device, dtype=torch.long), ts, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
            if ts in index_lst:
                img_lst.append(temp_out)

        reconstructed_image = vae.decode(temp_out).clamp(-1,1)
        rimage = reconstructed_image.squeeze(dim = 0).permute(1,2,0).cpu().numpy()
        rimage = (((rimage/2.0) + 0.5) * 255.0).astype(np.uint8)


        fig = plt.figure(figsize=(12, 12))
        columns = 3
        rows = 4
        for i in range(1, columns*rows +1):
            fig.add_subplot(rows, columns, i)
            reconstructed_image = vae.decode(img_lst[i-1]).clamp(-1,1)
            rimage = reconstructed_image.squeeze(dim = 0).permute(1,2,0).cpu().numpy()
            rimage = (((rimage/2.0) + 0.5) * 255.0).astype(np.uint8)
            plt.axis("off")
            plt.suptitle("Diffusion process over the timesteps (Read top left to bottom right) ")
            plt.imshow(rimage)
        # plt.show()
        plt.savefig("/home/tkarthikeyan/IIT_DELHI/COL775_Deep_Learning/2023AIZ8140_2023AIZ8073/diffusion_process/" + str(k) + ".png")
        

        k += 1
        
        if k == 10:
            break