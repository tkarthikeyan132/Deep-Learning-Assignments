import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ChainedScheduler, SequentialLR

from utils.dataset import CLEVERDataset
from utils.models import Encoder, Decoder
from utils.utilities import create_folder, store_json_file, plot_loss_graph, store_reconstructed_mask_image
from tqdm import tqdm

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is ", device)

# TO EDIT ZONE --------------------------------------------
#####Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0004
EPOCHS = 44
SAVE_STEPS = 4
NUM_WORKERS = 16

folder_dir = "/home/tkarthikeyan/IIT_DELHI/COL775_Deep_Learning/Assignment_2/dataset"
run_name = "44epochs_16batch/"
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

# for data in train_data_loader:
#     encoder = Encoder()
#     decoder = Decoder()
#     rimage, rimages, masks = decoder(encoder(data))
#     print(data.shape)
#     print(rimage.shape)
#     print(rimages.shape)
#     print(masks.shape)
#     break

def train():
    print("Inside training function ...")
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    # Define learning rate scheduler
    total_steps = len(train_data_loader) * EPOCHS
    print("Total steps: ", total_steps)

    scheduler1_e = LinearLR(encoder_optimizer, start_factor=0.0001, end_factor = 1, total_iters=2000)
    scheduler2_e = ExponentialLR(encoder_optimizer, gamma=0.99999)
    encoder_scheduler = SequentialLR(encoder_optimizer, schedulers=[scheduler1_e, scheduler2_e], milestones=[2000])

    scheduler1_d = LinearLR(decoder_optimizer, start_factor=0.001, end_factor = 1, total_iters=2000)
    scheduler2_d = ExponentialLR(decoder_optimizer, gamma=0.99999)
    decoder_scheduler = SequentialLR(decoder_optimizer, schedulers=[scheduler1_d, scheduler2_d], milestones=[2000])

    #Defining Loss
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0

        pbar = tqdm(train_data_loader, ncols = 120)
        pbar.set_postfix({"loss":100})
        
        for image in pbar:
            input_image_tensor = image.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            slots_tensor = encoder(input_image_tensor)
            output_image_tensor, output_images_tenosr, masks_tensor = decoder(slots_tensor)

            loss = criterion(output_image_tensor, input_image_tensor)
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

        # Validation loop
        # encoder.eval()
        # decoder.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_image in tqdm(val_data_loader):
                val_input_image_tensor = val_image.to(device)

                val_slots_tensor = encoder(val_input_image_tensor)
                val_output_image_tensor, val_output_images_tenosr, val_masks_tensor = decoder(val_slots_tensor)

                val_loss = criterion(val_output_image_tensor, val_input_image_tensor)

                if total_val_loss == 0:
                    store_reconstructed_mask_image(val_input_image_tensor, val_output_image_tensor, val_masks_tensor, output_dir, epoch)

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
