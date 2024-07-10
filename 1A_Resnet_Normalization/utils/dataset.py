import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image

class Birds25Dataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.bird_names = sorted(os.listdir(img_dir))
        self.bird_to_idx = {bird_name: idx for idx, bird_name in enumerate(self.bird_names)}
        self.idx_to_bird = {idx: bird_name for idx, bird_name in enumerate(self.bird_names)}
        
        self.images = self.make_dataset()
        self.n = len(self.images)
        
        self.transform = transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

    def make_dataset(self):
        images = []
        for bird_name in self.bird_names:
            bird_dir = os.path.join(self.img_dir, bird_name)
            for img_name in os.listdir(bird_dir):
                img_path = os.path.join(bird_dir, img_name)
                images.append((img_path, self.bird_to_idx[bird_name]))
        return images