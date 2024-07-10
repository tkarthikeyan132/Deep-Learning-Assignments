import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ChainedScheduler
from sklearn.cluster import KMeans
import random

from PIL import Image
from utils.dataset import CLEVERDataset
from utils.models import Encoder, Decoder
from utils.utilities import create_folder, store_json_file, plot_loss_graph
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is ", device)

# TO EDIT ZONE --------------------------------------------
#####Hyperparameters
BATCH_SIZE = 32

folder_dir = "/home/tkarthikeyan/IIT_DELHI/COL775_Deep_Learning/Assignment_2/dataset"
run_name = "batch64_400epochs/"
ckpt = "350" #"final"
#----------------------------------------------------------

checkpoint_dir = "checkpoints/" + run_name 
output_dir = "outputs/" + run_name + "val_new/"

encoder_checkpoint_path = os.path.join(checkpoint_dir, "encoder_" + ckpt+".pth") 
decoder_checkpoint_path = os.path.join(checkpoint_dir, "decoder_" + ckpt+".pth")

train_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "train")
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 16)

val_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "val")

encoder = Encoder().to(device)
decoder = Decoder().to(device)

encoder.load_state_dict(torch.load(encoder_checkpoint_path))
decoder.load_state_dict(torch.load(decoder_checkpoint_path))
print("model loaded!")

slots_list = []


with torch.no_grad():
    for data in tqdm(train_data_loader):
        image = data.to(device)
        
        slots = encoder(image)
        rimage, rimages, masks = decoder(slots)
        slots_reshaped = slots.reshape((-1,64))
        slots_list.append(slots_reshaped)


slots_combined = torch.stack(slots_list)
slots_combined = slots_combined.reshape((-1,64)).cpu()

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

        rimage, _, _ = decoder(sampled_slot_tensor)

        rimage = rimage[0].permute(1,2,0).cpu().numpy()
        rimage = (((rimage/2.0) + 0.5) * 255.0).astype(np.uint8)

    plt.imshow(rimage)
    plt.axis('off')  
    plt.savefig(output_dir + f'new_image_{kkk}.png')
    plt.clf()

