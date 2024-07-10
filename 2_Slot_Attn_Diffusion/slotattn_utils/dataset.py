import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image
import numpy as np

class CLEVERDataset(Dataset):
    def __init__(self, folder_dir, dataset_type="train", apply_crop = True):
        self.folder_dir = folder_dir
        self.apply_crop = apply_crop
        
        im_paths = sorted(os.listdir(os.path.join(folder_dir,"images", dataset_type)))
        self.image_paths = [os.path.join(folder_dir,"images", dataset_type, filename) for filename in im_paths]
        
        m_paths = sorted(os.listdir(os.path.join(folder_dir,"masks", dataset_type)))
        self.mask_paths = [os.path.join(folder_dir,"masks", dataset_type, filename) for filename in m_paths]
        
        self.n = len(self.image_paths)

        # self.transform = transforms.Compose([
            # transforms.CenterCrop((192, 192)),  # Center crop and returns a PIL image
            # transforms.Resize((128, 128)),  # Resize image to 128x128
            # transforms.ToTensor()  # Convert PIL image to tensor
        # ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        crop = (64, 29, 256, 221)
        image = image.crop(crop)
        
        image = image.resize((128, 128))
        
        # image = self.transform(image)
        image = torch.tensor(np.array(image))
        
        image = ((image / 255.0) - 0.5) * 2.0
        
        image = torch.clamp(image, -1., 1.)

        image = image.permute(2, 0, 1)
        return image

