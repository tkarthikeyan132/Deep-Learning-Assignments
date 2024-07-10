import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ChainedScheduler, SequentialLR, CosineAnnealingLR

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

from vae_ import VAE 

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is ", device)

# TO EDIT ZONE --------------------------------------------
#####Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
EPOCHS = 300
SAVE_STEPS = 5
NUM_WORKERS = 8

folder_dir = "/home/scai/phd/aiz238140/COL775/Assignment_2/A2_dataset/dataset"
run_name = "batch64_300epochs/"
#----------------------------------------------------------


checkpoint_dir = "checkpoints/" + run_name 
output_dir = "outputs/" + run_name 

create_folder(output_dir)
create_folder(checkpoint_dir)
train_loss_list, val_loss_list = [], []

train_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "train")
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)

val_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "val")
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)



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

def train():
    print("Inside training function ...")
    encoder = Encoder().to(device)
    decoder = UNET().to(device)

    random_timestep = random.randint(0, 1000)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    # Define learning rate scheduler
    total_steps = len(train_data_loader) * EPOCHS
    print("Total steps: ", total_steps)

    # scheduler1_e = LinearLR(encoder_optimizer, start_factor=0.0001, end_factor = 1, total_iters=2000)
    # scheduler2_e = ExponentialLR(encoder_optimizer, gamma=0.99999)
    # encoder_scheduler = SequentialLR(encoder_optimizer, schedulers=[scheduler1_e, scheduler2_e], milestones=[2000])

    # scheduler1_d = LinearLR(decoder_optimizer, start_factor=0.001, end_factor = 1, total_iters=2000)
    # scheduler2_d = ExponentialLR(decoder_optimizer, gamma=0.99999)
    # decoder_scheduler = SequentialLR(decoder_optimizer, schedulers=[scheduler1_d, scheduler2_d], milestones=[2000])

    encoder_scheduler = CosineAnnealingLR(encoder_optimizer, T_max=total_steps)
    decoder_scheduler = CosineAnnealingLR(decoder_optimizer, T_max=total_steps)

    #Defining Loss
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0

        pbar = tqdm(train_data_loader, ncols = 75)
        pbar.set_postfix({"loss":100})
        
        for image in pbar:
            input_image_tensor = image.to(device)
            vae_out = vae.encode(input_image_tensor)
            
            t = torch.randint(0, 1000, (vae_out.shape[0], ), device=device).long()
        
            noise = torch.randn_like(vae_out)
            x_noisy = q_sample(vae_out, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            slots_tensor, attn_output = encoder(input_image_tensor)
            
            sinusoidal_timestep = SP(t).to(device)
            
            unet_out = decoder(x_noisy, sinusoidal_timestep, slots_tensor)

            loss = criterion(unet_out, noise)
            total_loss += loss.item()

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            # print(f"Learning rate:",encoder_optimizer.param_groups[0]["lr"])
            encoder_scheduler.step()
            decoder_scheduler.step()

            pbar.set_postfix({"loss":loss.item()})
            pbar.update()

        print(f"Epoch {epoch}/{EPOCHS}: Train Loss:{total_loss / len(train_data_loader)}") 
        train_loss_list.append(total_loss / len(train_data_loader))

        with torch.no_grad():
            total_val_loss = 0
            for val_image in tqdm(val_data_loader):
                val_input_image_tensor = val_image.to(device)
                vae_out = vae.encode(val_input_image_tensor)

                t = torch.randint(0, 1000, (vae_out.shape[0], ), device=device).long()
                noise = torch.randn_like(vae_out)
                x_noisy = q_sample(vae_out, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)

                val_slots_tensor, val_attn_output = encoder(val_input_image_tensor)
                sinusoidal_timestep = SP(t).to(device)

                unet_out = decoder(x_noisy, sinusoidal_timestep, val_slots_tensor)

                val_loss = criterion(unet_out, noise)

                if total_val_loss == 0:
                    store_reconstructed_mask_image(val_input_image_tensor, vae.decode(unet_out), F.interpolate(val_attn_output.reshape((val_input_image_tensor.shape[0],11,32,32)),size=(128,128), mode='bilinear', align_corners=False), output_dir, epoch)

                total_val_loss += val_loss.item()

            val_loss = total_val_loss / len(val_data_loader)
            print(f"Epoch {epoch}/{EPOCHS}: Validation Loss:{val_loss}")
            val_loss_list.append(val_loss)

        if (epoch%SAVE_STEPS == 0):
            torch.save(encoder.state_dict(), checkpoint_dir + "encoder_" + str(epoch) + ".pth")
            torch.save(decoder.state_dict(), checkpoint_dir + "decoder_" + str(epoch) + ".pth")

            print(f"Model saved!!!")

        print(f"Learning rate:{encoder_scheduler.get_last_lr()}")

    print("Training complete!")
    torch.save(encoder.state_dict(), checkpoint_dir + "encoder_final.pth")
    torch.save(decoder.state_dict(), checkpoint_dir + "decoder_final.pth")
    print(f"Model saved!!!")

    epochs = [i for i in range(1,EPOCHS+1)]
    plot_loss_graph(train_loss_list, val_loss_list, epochs, output_dir)

    store_json_file(train_loss_list, val_loss_list, BATCH_SIZE, LEARNING_RATE, EPOCHS, output_dir)

def main():    
    train()

if __name__ == "__main__":
    main()
