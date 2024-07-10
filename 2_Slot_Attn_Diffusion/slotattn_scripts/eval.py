import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ChainedScheduler

from PIL import Image
from utils.dataset import CLEVERDataset
from utils.models import Encoder, Decoder
from utils.utilities import create_folder, store_json_file, plot_loss_graph
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

def preprocess_mask(img_path):
    image = Image.open(img_path).convert("RGB")
    
    arr = np.array(image)
    
    # Convert RGB mask to one-dimensional mask with unique integer values
    unique_mask, unique_indices = np.unique(arr.reshape(-1, 3), axis=0, return_inverse=True)
    one_dim_mask = unique_indices.reshape(arr.shape[:2])

    # Convert the one-dimensional mask to uint8 type
    one_dim_mask_uint8 = one_dim_mask.astype(np.uint8)
    image = Image.fromarray(one_dim_mask_uint8)
    
    crop = (64, 29, 256, 221)
    image = image.crop(crop)
    
    image = image.resize((128, 128))
    
    image = torch.tensor(np.array(image))
    
    image = image.unsqueeze(dim = 0)
    return image

def adjusted_rand_index(true_masks, pred_masks):
    """
    Args:
        true_masks: Integer ids for objects
            [batch_size, H, W].  
            as integer ids.
        pred_masks: An integer-valued array of shape
            [batch_size, K, H, W]. The predicted cluster assignment
            encoded as integer ids.
        ignore_background: Boolean, if True, then ignore all pixels where
            true_ids == 0 (default: False).

    Returns:
        ARI scores as a float32 array of shape [batch_size].
    """
    pred_masks = pred_masks.argmax(dim=-3)  # [B, N, H, W] --> [B, H, W]

    # if len(true_ids.shape) == 3:
        # true_ids = true_ids.unsqueeze(1)
    # if len(pred_ids.shape) == 3:
        # pred_ids = pred_ids.unsqueeze(1)

    if len(true_masks.shape) == 3:
        true_ids = true_masks.unsqueeze(1)
    if len(pred_masks.shape) == 3:
        pred_ids = pred_masks.unsqueeze(1)

    # true_oh = F.one_hot(true_ids).float()
    # pred_oh = F.one_hot(pred_ids).float()

    true_oh = F.one_hot(true_ids.to(torch.int64)).float()
    pred_oh = F.one_hot(pred_ids.to(torch.int64)).float()

    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = torch.sum(N, dim=-1)  # row-sum  (batch_size, c)
    B = torch.sum(N, dim=-2)  # col-sum  (batch_size, k)
    num_points = torch.sum(A, dim=1)

    rindex = torch.sum(N * (N - 1), dim=[1, 2])
    aindex = torch.sum(A * (A - 1), dim=1)
    bindex = torch.sum(B * (B - 1), dim=1)
    expected_rindex = aindex * bindex / torch.clamp(
        num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator != 0, ari, torch.tensor(1.).type_as(ari))


#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is ", device)

# TO EDIT ZONE --------------------------------------------
#####Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 0.0004
EPOCHS = 2
SAVE_STEPS = 1

folder_dir = "/home/tkarthikeyan/IIT_DELHI/COL775_Deep_Learning/Assignment_2/dataset"
run_name = "batch64_400epochs/"
ckpt = "350" #"final"
#----------------------------------------------------------

checkpoint_dir = "checkpoints/" + run_name 
output_dir = "outputs/" + run_name 

encoder_checkpoint_path = os.path.join(checkpoint_dir, "encoder_" + ckpt+".pth") 
decoder_checkpoint_path = os.path.join(checkpoint_dir, "decoder_" + ckpt+".pth")

# train_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "train")
# train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 16)

val_dataset = CLEVERDataset(folder_dir=folder_dir, dataset_type = "val")
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 16)

mask_paths = [os.path.join(folder_dir,"masks", "val/") + l for l in sorted(os.listdir(os.path.join(folder_dir,"masks", "val")))]

criterion = nn.MSELoss()

encoder = Encoder().to(device)
decoder = Decoder().to(device)

encoder.load_state_dict(torch.load(encoder_checkpoint_path))
decoder.load_state_dict(torch.load(decoder_checkpoint_path))
print("model loaded!")

ari_score = 0
zz = 0
with torch.no_grad():
    for data in tqdm(val_data_loader):
        image = data.to(device)
        
        slots = encoder(image)
        rimage, rimages, masks = decoder(slots)
        
        # print(criterion(rimage[0], image[0]).item())
        pred_masks = masks.squeeze(dim = 4).to(device)
        true_mask = preprocess_mask(mask_paths[zz]).to(device)
        # print(true_mask.unique())
        
        ari_score += adjusted_rand_index(true_mask, pred_masks).item()

        # img1 = (image[0].permute(1,2,0).cpu().numpy())
        # fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,14))
        # ax1.imshow(img1)
        # ax1.set_title('Real image')
        # plt.axis("off")

        # img2 = (rimage[0].permute(1,2,0).cpu().numpy())
        # ax2.imshow(img2)
        # ax2.set_title('Reconstructed image')
        # plt.title(correct_entry[1][0] + "_predicted_as_" + correct_entry[2][0])
        # plt.tight_layout()
        # plt.axis("off")
        # plt.show()
        # print(data.shape)
        # print(rimage.shape)
        # print(rimages.shape)
        # print(masks.shape)
        zz += 1

ari_score /= len(val_data_loader)
print("Validation ARI score is ", ari_score)