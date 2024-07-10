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
run_name = "batch64_300epochs/"
ckpt = "final"
#----------------------------------------------------------


checkpoint_dir = "checkpoints/" + run_name 
output_dir = "outputs/" + run_name + "val_new/"

create_folder(output_dir)

encoder_checkpoint_path = os.path.join(checkpoint_dir, "encoder_" + ckpt+".pth") 
decoder_checkpoint_path = os.path.join(checkpoint_dir, "decoder_" + ckpt+".pth")

train_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "train")
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)

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

val_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "val")

encoder = Encoder().to(device)
decoder = UNET().to(device)

encoder.load_state_dict(torch.load(encoder_checkpoint_path))
decoder.load_state_dict(torch.load(decoder_checkpoint_path))
print("model loaded!")

slots_list = []

with torch.no_grad():
    for data in tqdm(train_data_loader):
        image = data.to(device)
        
        slots, _ = encoder(image)
        # rimage, rimages, masks = decoder(slots)
        slots_reshaped = slots.reshape((-1,192))
        slots_list.append(slots_reshaped)

slots_combined = torch.stack(slots_list)
slots_combined = slots_combined.reshape((-1,192)).cpu()

kmeans = KMeans(n_clusters=11, random_state=0, n_init="auto").fit(slots_combined)
labels = kmeans.labels_

slots_groupwise = [[] for _ in range(11)]

for i in range(slots_combined.shape[0]):
    slots_groupwise[labels[i]].append(slots_combined[i])


for kkk in tqdm(range(len(val_dataset))):
    sampled_slot = [random.sample(slot_i, 1)[0] for slot_i in slots_groupwise]

    # print(sampled_slot[0])
    with torch.no_grad():
        sampled_slot_tensor = torch.stack(sampled_slot).unsqueeze(dim = 0).to(device)

        #Initial normal sampled image tensor (X^T)
        temp_out = torch.randn((1, 3, 32, 32)).to(device)

        for ts in tqdm(reversed(range(0, 1000))):
            sinusoidal_timestep = SP(torch.tensor([ts])).to(device)
        
            unet_out = decoder(temp_out, sinusoidal_timestep, sampled_slot_tensor)

            temp_out = p_sample(unet_out, temp_out, torch.full((1,), ts, device=device, dtype=torch.long), ts, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)

        reconstructed_image = vae.decode(temp_out).clamp(-1,1)
        rimage = reconstructed_image.squeeze(dim = 0).permute(1,2,0).cpu().numpy()
        rimage = (((rimage/2.0) + 0.5) * 255.0).astype(np.uint8)

    plt.imshow(rimage)
    plt.axis('off')  
    plt.savefig(output_dir + f'new_image_{kkk+1470}.png')
    plt.clf()

    
