#Libraries
import argparse
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchview import draw_graph

from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import random

import graphviz
graphviz.set_jupyter_format('png')

from utils.dataset import Birds25Dataset
from utils.evaluate import evaluate_model, create_folder
from utils.plot import all_plots

idx_to_bird = {0: 'Asian-Green-Bee-Eater', 1: 'Brown-Headed-Barbet', 2: 'Cattle-Egret', 3: 'Common-Kingfisher', 4: 'Common-Myna', 5: 'Common-Rosefinch', 6: 'Common-Tailorbird', 7: 'Coppersmith-Barbet', 8: 'Forest-Wagtail', 9: 'Gray-Wagtail', 10: 'Hoopoe', 11: 'House-Crow', 12: 'Indian-Grey-Hornbill', 13: 'Indian-Peacock', 14: 'Indian-Pitta', 15: 'Indian-Roller', 16: 'Jungle-Babbler', 17: 'Northern-Lapwing', 18: 'Red-Wattled-Lapwing', 19: 'Ruddy-Shelduck', 20: 'Rufous-Treepie', 21: 'Sarus-Crane', 22: 'White-Breasted-Kingfisher', 23: 'White-Breasted-Waterhen', 24: 'White-Wagtail'}

def sort_key(file_name):
    return int(file_name.split('.')[0])

def convert_img_to_tensor(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(dim=0)


def main():
    parser = argparse.ArgumentParser(description='Process inference parameters')
    parser.add_argument('--model_file', type=str, help='Path to the trained model')
    parser.add_argument('--normalization', type=str, choices=['bn', 'in', 'bin', 'ln', 'gn', 'nn', 'inbuilt'], help='Normalization method')
    parser.add_argument('--n', type=int, choices=[1, 2, 3], help='Number parameter')
    parser.add_argument('--test_data_file', type=str, help='Path to the directory containing the images')
    parser.add_argument('--output_file', type=str, help='File containing the prediction in the same order as the images in directory')
    
    args = parser.parse_args()

    if args.normalization == 'inbuilt':
        print("inbuilt bn resnet loaded!")
        from model_defs.resnet_bn_inbuilt import ResNet
    elif args.normalization == 'bn':
        print("bn resnet loaded!")
        from model_defs.resnet_bn_my import ResNet 
    elif args.normalization == 'ln':
        print("ln resnet loaded!")
        from model_defs.resnet_ln_my import ResNet 
    elif args.normalization == 'in':
        print("in resnet loaded!")
        from model_defs.resnet_in_my import ResNet
    elif args.normalization == 'bin':
        print("bin resnet loaded!")
        from model_defs.resnet_bin_my import ResNet
    elif args.normalization == 'gn':
        print("gn resnet loaded!")
        from model_defs.resnet_gn_my import ResNet
    elif args.normalization == 'nn':
        print("nn resnet loaded!")
        from model_defs.resnet_nn_my import ResNet

    #Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is ", device)

    #HYPERPARAMETERS
    BATCH_SIZE=32
    LEARNING_RATE = 0.0001
    N = args.n
    R = 25

    #PARAMETERS
    checkpoint_path = args.model_file
    
    img_dir = args.test_data_file

    output_file_path = args.output_file

    images = os.listdir(img_dir)
    images = sorted(images, key=sort_key)
    # print(images)
    
    images = [os.path.join(img_dir, image) for image in images]
    images = [convert_img_to_tensor(image) for image in images] #images contain list of tensors
    
    predictions = []

    #Declaring the model
    net = ResNet(n = N, r = R).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    #Loading the model
    net.load_state_dict(torch.load(checkpoint_path))
    print(f"model at {checkpoint_path} loaded!")
    
    with torch.no_grad():
        for x in images:
            inputs = x.to(device)
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(idx_to_bird[preds.item()])

    # Writing the predictions to output file
    with open(output_file_path, 'w') as file:
        for item in predictions:
            file.write("%s\n" % item)

    print("Output written to", output_file_path)

if __name__ == "__main__":
    main()
