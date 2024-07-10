#Libraries
import json 
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext.vocab import GloVe

from tqdm import tqdm
import numpy as np
plt.switch_backend('agg')

from utils import *
from dataset import *

#Arch 1: BiLSTM + LSTM Decoder
# from model import *

#Arch 2: BiLSTM + Attn LSTM Decoder
from model_attn import *

#Checking the conda environment
print(os.environ['CONDA_DEFAULT_ENV'])

#Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####Hyperparameters
HIDDEN_DIM = 128
BATCH_SIZE = 16
EMBED_DIM = 100
TEACHER_FORCING = 0.3
BEAM_WIDTH = 1

#####Parameters
LEARNING_RATE = 0.001
EPOCHS = 120
SAVE_STEPS = 10

# Define special tokens
SOS_token = '<sos>'
EOS_token = '<eos>'
PAD_token = '<pad>'
UNK_token = '<unk>'

train_file_path = "/home/tkarthikeyan/IIT_Delhi/COL775_Deep_Learning/Assignment_1b/data/train.json"
val_file_path = "/home/tkarthikeyan/IIT_Delhi/COL775_Deep_Learning/Assignment_1b/data/dev.json"
run_name = "arch2_120epochs_tf_point3/"


# Create a new vocabulary with special tokens
special_tokens = [SOS_token, EOS_token, PAD_token, UNK_token]

glove_vector = GloVe(name='6B', dim=100)
special_embeddings = torch.load("special_embeddings_glove_100.pt")

train_loss_list, val_loss_list = [], []

#Loading Training data and making train dataloader
with open(train_file_path, "r") as file:
    train_data = json.load(file)

with open(val_file_path, "r") as file:
    val_data = json.load(file)

problem_vocab, formula_vocab = build_input_vocab(train_data)

train_dataset = ProblemFormulaDataset(train_data, problem_vocab, glove_vector, special_embeddings, formula_vocab)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

val_dataset = ProblemFormulaDataset(val_data, problem_vocab, glove_vector, special_embeddings, formula_vocab)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


OUTPUT_DIM = len(formula_vocab)

output_dir = "outputs/" + run_name
checkpoint_dir = "checkpoints/" + run_name 

create_folder(output_dir)
create_folder(checkpoint_dir)
# for problem, linear_formula in train_dataloader:
#     print(problem.shape)
#     print(linear_formula.shape)
#     break

def train():
    print("Inside training mode ...")
    encoder = EncoderRNN(EMBED_DIM, HIDDEN_DIM).to(device)
    decoder = DecoderRNN(2*HIDDEN_DIM, OUTPUT_DIM, teacher_forcing_ratio=TEACHER_FORCING, beam_width=BEAM_WIDTH).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    encoder_scheduler = ReduceLROnPlateau(encoder_optimizer, 'min', verbose=True)
    decoder_scheduler = ReduceLROnPlateau(decoder_optimizer, 'min', verbose=True)

    criterion = nn.NLLLoss()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0

        for problem, linear_formula in tqdm(train_dataloader):
            input_tensor, target_tensor = problem.to(device), linear_formula.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)
            decoder_outputs, _, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_cell, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS}: Train Loss:{total_loss / len(train_dataloader)}") 
        train_loss_list.append(total_loss / len(train_dataloader))

        # Validation loop
        # encoder.eval()
        # decoder.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_problem, val_linear_formula in tqdm(val_dataloader):
                val_input_tensor, val_target_tensor = val_problem.to(device), val_linear_formula.to(device)

                val_encoder_outputs, val_encoder_hidden, val_encoder_cell = encoder(val_input_tensor)
                val_decoder_outputs, _, _, _ = decoder(val_encoder_outputs, val_encoder_hidden, val_encoder_cell, val_target_tensor)

                val_loss = criterion(
                    val_decoder_outputs.view(-1, val_decoder_outputs.size(-1)),
                    val_target_tensor.view(-1)
                )
                total_val_loss += val_loss.item()

            val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch}/{EPOCHS}: Validation Loss:{val_loss}")
            val_loss_list.append(val_loss)

        if (epoch%SAVE_STEPS == 0):
            torch.save(encoder.state_dict(), checkpoint_dir + "encoder_" + str(epoch) + ".pth")
            torch.save(decoder.state_dict(), checkpoint_dir + "decoder_" + str(epoch) + ".pth")

            file_path = "/home/tkarthikeyan/IIT_Delhi/COL775_Deep_Learning/Assignment_1b/data/"
            test(file_path=file_path + "dev.json", store_file_path=output_dir + f"dev_predicted_{epoch}.json", instance=epoch)
            test(file_path=file_path + "test.json", store_file_path=output_dir + f"test_predicted_{epoch}.json", instance=epoch)

            print(f"Model saved!!!")

        encoder_scheduler.step(val_loss)
        decoder_scheduler.step(val_loss)

        print(f"Learning rate:{encoder_scheduler.get_last_lr()}")

    print("Training complete!")
    torch.save(encoder.state_dict(), checkpoint_dir + "encoder_final.pth")
    torch.save(decoder.state_dict(), checkpoint_dir + "decoder_final.pth")
    print(f"Model saved!!!")

    print(f"dev predicted and val predicted jsons created!")

    epochs = [i for i in range(1,EPOCHS+1)]
    plot_loss_graph(train_loss_list, val_loss_list, epochs, output_dir)

    store_json_file(train_loss_list, val_loss_list, HIDDEN_DIM, BATCH_SIZE, EMBED_DIM, LEARNING_RATE, EPOCHS, TEACHER_FORCING, output_dir)

def test(file_path, store_file_path, instance):
    print("Inside testing mode ...")
    
    with open(file_path, "r") as file:
        test_data = json.load(file)
    
    pred_data = []

    encoder = EncoderRNN(EMBED_DIM, HIDDEN_DIM).to(device)
    decoder = DecoderRNN(2*HIDDEN_DIM, OUTPUT_DIM, teacher_forcing_ratio=TEACHER_FORCING, beam_width=BEAM_WIDTH).to(device)
    encoder.load_state_dict(torch.load(checkpoint_dir + f"encoder_{instance}.pth"))
    decoder.load_state_dict(torch.load(checkpoint_dir + f"decoder_{instance}.pth"))

    encoder.eval()
    decoder.eval()

    for data in tqdm(test_data):
        # question = "the denominator of a fraction is 6 greater than the numerator . if the numerator and the denominator are increased by 1 , the resulting fraction is equal to 4 \u00e2 \u0081 \u201e 5 . what is the value of the original fraction ?"
        question = data["Problem"]

        input_tensor = train_dataset.preprocess_problem(question).unsqueeze(0).to(device)

        encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)
        _, decoded_ids, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_cell, None)

        # _, topi = decoder_outputs.topk(1)
        # decoded_ids = topi.squeeze()

        ans = decode_to_string(decoded_ids, formula_vocab)

        temp_data = data.copy()
        temp_data["predicted"] = ans.replace("<sos>","").replace("<eos>","")
        pred_data.append(temp_data)

    with open(store_file_path, "w") as wfile:
        json.dump(pred_data, wfile, indent=4)

    print("Testing complete!!!")
    print(f"predictions stored in {store_file_path}")

def main():    
    train()

    file_path = "/home/tkarthikeyan/IIT_Delhi/COL775_Deep_Learning/Assignment_1b/data/"
    instance = "final"
    # instance = 6
    test(file_path=file_path + "dev.json", store_file_path=output_dir + f"dev_predicted_{instance}.json", instance=instance)

    test(file_path=file_path + "test.json", store_file_path=output_dir + f"test_predicted_{instance}.json", instance=instance)

    print("Dev data:")
    os.system("python data/evaluator.py " + output_dir + f"dev_predicted_{instance}.json")

    print("Test data:")
    os.system("python data/evaluator.py " + output_dir + f"test_predicted_{instance}.json")

if __name__ == "__main__":
    main()