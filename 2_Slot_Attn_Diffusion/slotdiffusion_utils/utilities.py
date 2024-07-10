import os
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from vae_ import VAE 

# #Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# vae = VAE().to(device)
# ckpt = torch.load("vae_checkpoint.pth")
# vae.load_state_dict(ckpt)

def store_reconstructed_mask_image(val_input_image_tensor, val_output_image_tensor, val_masks_tensor, output_dir, epoch = 9999999):
    mlst = []
    lst = []
    lst.append(val_input_image_tensor[0].permute(1,2,0).cpu().numpy())
    for k in range(11):
        lst.append(val_masks_tensor[0][k].cpu().numpy())
    lst.append(val_output_image_tensor[0].permute(1,2,0).cpu().numpy())
    mlst.append(lst)

    lst = []
    lst.append(val_input_image_tensor[1].permute(1,2,0).cpu().numpy())
    for k in range(11):
        lst.append(val_masks_tensor[1][k].cpu().numpy())
    lst.append(val_output_image_tensor[1].permute(1,2,0).cpu().numpy())
    mlst.append(lst)

    # Create a 2x13 grid
    fig, axs = plt.subplots(2, 13, figsize=(52, 20))

    for i in range(2):
        for j in range(13):
            ax = axs[i, j]
            if j > 0 and j < 12:
                ax.imshow(mlst[i][j], cmap="gray")
            else:
                ax.imshow(mlst[i][j])
            ax.axis('off')  # Turn off axis


    plt.title(f"Recons image after epoch {str(epoch)}")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(output_dir + f'validation_image.png')

def store_json_file(train_loss_list, val_loss_list, BATCH_SIZE, LEARNING_RATE, EPOCHS, output_dir):
    result = {
        "final_train loss": train_loss_list[-1],
        "final_val loss": val_loss_list[-1],
        "train_loss_list":train_loss_list,
        "val_loss_list":val_loss_list,
        "Hyperparameters":{
            "BATCH SIZE": BATCH_SIZE,
            "LEARNING RATE": LEARNING_RATE,
            "EPOCHS": EPOCHS
        }
    }

    # Store results in a JSON file
    with open(output_dir + "results.json", "w") as json_file:
        json.dump(result, json_file, indent=4)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def plot_loss_graph(train_loss_list, val_loss_list, epochs, output_dir):
    #Loss vs epoch graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig(output_dir + 'loss_plot.png')