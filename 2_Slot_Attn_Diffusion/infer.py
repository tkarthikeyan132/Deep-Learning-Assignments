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

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import random
import os
import argparse

from vae_ import VAE 

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is ", device)

def store_all_plots_slotattn(rimage, attn_masks, img_name, output_dir):
    plt.imshow(rimage[0].permute(1, 2, 0).cpu().numpy())  # Plot the color image
    plt.axis('off')  
    plt.savefig(os.path.join(output_dir, img_name))  # Save the image to a file
    plt.close()

    img_name_name = img_name.split(".")[0]
    for i in range(attn_masks.shape[1]):
        temp_img = attn_masks[0][i].cpu().numpy()
        plt.imshow(temp_img, cmap="gray")  # Plot the color image
        plt.axis('off')  
        plt.savefig(os.path.join(output_dir, img_name_name + '_' + str(i) + '.png'))  # Save the image to a file
        plt.close()

def store_all_plots_slotdiffusion(rimage, attn_masks, img_name, output_dir):
    plt.imshow(rimage)  # Plot the color image
    plt.axis('off')  
    plt.savefig(os.path.join(output_dir, img_name))  # Save the image to a file
    plt.close()

    img_name_name = img_name.split(".")[0]
    for i in range(attn_masks.shape[1]):
        temp_img = attn_masks[0][i].unsqueeze(dim = 2).cpu().numpy()
        plt.imshow(temp_img, cmap="gray")  # Plot the color image
        plt.axis('off')  
        plt.savefig(os.path.join(output_dir, img_name_name + '_' + str(i) + '.png'))  # Save the image to a file
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Process inference parameters')
    
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing images')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--part', type=int, choices=[1,2], help='1 for Slot Attention and 2 for Slot Diffusion')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory where results will be stored')
    
    args = parser.parse_args()

    img_list = sorted(os.listdir(args.input_dir))
    img_paths = [ os.path.join(args.input_dir, img) for img in img_list]

    if args.part == 1:
        print("Slot Attention mode ...")
        from slotattn_utils.dataset import CLEVERDataset
        from slotattn_utils.models import Encoder, Decoder
        from slotattn_utils.utilities import create_folder, store_json_file, plot_loss_graph

        create_folder(args.output_dir)

        encoder = Encoder().to(device)
        decoder = Decoder().to(device)

        model = torch.load(args.model_path)
        encoder.load_state_dict(model["encoder"])
        decoder.load_state_dict(model["decoder"])
        print("slot attention model loaded!")

        with torch.no_grad():
            for img_path in tqdm(img_paths):
                img_name = img_path.split("/")[-1]
                image = Image.open(img_path).convert("RGB")
                
                crop = (64, 29, 256, 221)
                image = image.crop(crop)
                image = image.resize((128, 128))
                image = torch.tensor(np.array(image))
                image = ((image / 255.0) - 0.5) * 2.0
                image = torch.clamp(image, -1., 1.)
                image = image.permute(2, 0, 1)
                image = image.to(device)
                image = image.unsqueeze(0)
        
                slots = encoder(image)
                rimage, rimages, masks = decoder(slots)

                store_all_plots_slotattn(rimage, masks, img_name, args.output_dir)

    if args.part == 2:
        print("Slot Diffusion mode ...")   
        from slotdiffusion_utils.dataset import CLEVERDataset
        from slotdiffusion_utils.models import Encoder, Decoder
        from slotdiffusion_utils.models_unet import ResBlock, SpatialTransformer, UNET, SinusoidalPositionEmbeddings
        from slotdiffusion_utils.utilities import create_folder, store_json_file, plot_loss_graph, store_reconstructed_mask_image
        from slotdiffusion_utils.diffusion_utils import extract, linear_beta_schedule, q_sample, p_sample

        create_folder(args.output_dir)

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

        model = torch.load(args.model_path)
        encoder.load_state_dict(model["encoder"])
        decoder.load_state_dict(model["decoder"])
        print("slot diffusion model loaded!")

        with torch.no_grad():
            for img_path in tqdm(img_paths):
                img_name = img_path.split("/")[-1]
                image = Image.open(img_path).convert("RGB")
                
                crop = (64, 29, 256, 221)
                image = image.crop(crop)
                image = image.resize((128, 128))
                image = torch.tensor(np.array(image))
                image = ((image / 255.0) - 0.5) * 2.0
                image = torch.clamp(image, -1., 1.)                
                image = image.permute(2, 0, 1)
                image = image.to(device)
                image = image.unsqueeze(0)
                slots, attn_masks = encoder(image)
                attn_masks = attn_masks.reshape((1,11,32,32))
                rescaled_attn_masks = F.interpolate(attn_masks, size=(128,128), mode='bilinear', align_corners=False)

                temp_out = torch.randn((1, 3, 32, 32)).to(device)

                for ts in tqdm(reversed(range(0, 1000))):
                    sinusoidal_timestep = SP(torch.tensor([ts])).to(device)
                    unet_out = decoder(temp_out, sinusoidal_timestep, slots)
                    temp_out = p_sample(unet_out, temp_out, torch.full((1,), ts, device=device, dtype=torch.long), ts, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)

                reconstructed_image = vae.decode(temp_out).clamp(-1,1)
                rimage = reconstructed_image.squeeze(dim = 0).permute(1,2,0).cpu().numpy()
                rimage = (((rimage/2.0) + 0.5) * 255.0).astype(np.uint8)

                store_all_plots_slotdiffusion(rimage, rescaled_attn_masks, img_name, args.output_dir)

if __name__ == "__main__":
    main()