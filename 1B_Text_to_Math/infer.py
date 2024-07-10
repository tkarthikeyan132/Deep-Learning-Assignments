#Libraries
import argparse
import json 
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext.vocab import GloVe
from transformers import BertModel, BertTokenizer

from tqdm import tqdm
import numpy as np
plt.switch_backend('agg')

from utils import *
from dataset import *
import pickle

def preprocess_problem(question, special_embeddings, glove_vector):
    masked_question = get_mask_question(question)
    question_toks = masked_question.split()

    embeddings = []
    embeddings.append(torch.tensor(special_embeddings['<sos>'], dtype=torch.float32))
    for word in question_toks:
        if word in glove_vector.stoi:
            embeddings.append(torch.tensor(glove_vector[word], dtype=torch.float32))
        else:
            embeddings.append(torch.tensor(special_embeddings['<unk>'], dtype=torch.float32))

    embeddings.append(torch.tensor(special_embeddings['<eos>'], dtype=torch.float32))

    return torch.stack(embeddings)

def preprocess_problem_bert(question, tokenizer, max_length = 128):
    masked_question = get_mask_question(question)
    encoding = tokenizer(masked_question, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].squeeze(0)
    attention_mask = encoding['attention_mask'].squeeze(0)
    return input_ids, attention_mask

def main():
    parser = argparse.ArgumentParser(description='Process inference parameters')
    parser.add_argument('--model_file', type=str, help='Path to the trained model')
    parser.add_argument('--beam_size', type=int, choices=[1, 10, 20], help='Beam size for beam search')
    parser.add_argument('--model_type', type=str, choices=['lstm_lstm', 'lstm_lstm_attn', 'bert_lstm_attn_frozen', 'bert_lstm_attn_tuned'], help='Type of the trained model')
    parser.add_argument('--test_data_file', type=str, help='Path to the test data file')
    
    args = parser.parse_args()

    if args.model_type == 'lstm_lstm':
        print("lstm lstm functions loaded!")
        from model_defs.model import EncoderRNN, DecoderRNN
        HIDDEN_DIM = 128
    elif args.model_type == 'lstm_lstm_attn':
        print("lstm lstm attn functions loaded!")
        from model_defs.model_attn import EncoderRNN, BahdanauAttention, DecoderRNN
        HIDDEN_DIM = 128
    elif args.model_type == 'bert_lstm_attn_frozen':
        print("bert lstm attn frozen functions loaded!")
        from model_defs.model_bert_attn import EncoderRNN, BahdanauAttention, DecoderRNN
        HIDDEN_DIM = 768
    elif args.model_type == 'bert_lstm_attn_tuned':
        print("bert lstm attn tuned functions loaded!")
        from model_defs.model_bert_attn_unfreeze import EncoderRNN, BahdanauAttention, DecoderRNN
        HIDDEN_DIM = 768

    #Setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####Hyperparameters
    EMBED_DIM = 100
    TEACHER_FORCING = 0.6 #Ignored during inference but needed for initialization
    BEAM_WIDTH = args.beam_size

    #####Parameters (No use in inference)
    LEARNING_RATE = 0.001
    EPOCHS = 120
    SAVE_STEPS = 10

    file_path = args.test_data_file
    beam_size = args.beam_size
    checkpoint_dir = args.model_file
    # encoder_checkpoint = "models/encoder_final.pth"
    # decoder_checkpoint = "models/decoder_final.pth" 

    
    #Loading data 
    with open(file_path, "r") as file:
        data = json.load(file)

    # problem_vocab, formula_vocab = build_input_vocab(train_data)
    with open("problem_vocab.pkl", 'rb') as file:
        problem_vocab = pickle.load(file)

    with open("formula_vocab.pkl", 'rb') as file:
        formula_vocab = pickle.load(file)

    print(len(problem_vocab))
    print(len(formula_vocab))

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    glove_vector = GloVe(name='6B', dim=100)
    special_embeddings = torch.load("special_embeddings_glove_100.pt")

    OUTPUT_DIM = len(formula_vocab)

    if args.model_type in ['lstm_lstm', 'lstm_lstm_attn']:
        encoder = EncoderRNN(EMBED_DIM, HIDDEN_DIM).to(device)
        decoder = DecoderRNN(2*HIDDEN_DIM, OUTPUT_DIM, teacher_forcing_ratio=TEACHER_FORCING, beam_width=BEAM_WIDTH).to(device)
    elif args.model_type in ['bert_lstm_attn_frozen', 'bert_lstm_attn_tuned']:
        encoder = EncoderRNN().to(device)
        decoder = DecoderRNN(HIDDEN_DIM, OUTPUT_DIM, teacher_forcing_ratio=TEACHER_FORCING, beam_width=BEAM_WIDTH).to(device)

    # encoder.load_state_dict(torch.load(encoder_checkpoint))
    # decoder.load_state_dict(torch.load(decoder_checkpoint))
    model = torch.load(checkpoint_dir)
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"])

    # combined_state_dict = {}
    # combined_state_dict["encoder"] = encoder.state_dict()
    # combined_state_dict["decoder"] = decoder.state_dict()
    # torch.save(combined_state_dict, "models/lstm_lstm_attn.pth")

    encoder.eval()
    decoder.eval()
    print("model loaded!")

    new_data = []

    if args.model_type in ['bert_lstm_attn_frozen', 'bert_lstm_attn_tuned']:
        with torch.no_grad():
            for x in data:
                question = x["Problem"]
                input_tensor, attn_tensor = preprocess_problem_bert(question, bert_tokenizer)
                input_tensor = input_tensor.unsqueeze(dim=0).to(device)
                attn_tensor = attn_tensor.unsqueeze(dim=0).to(device)

                encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor, attn_tensor)
                _, decoded_ids, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_cell, None)

                # _, topi = decoder_outputs.topk(1)
                # decoded_ids = topi.squeeze()

                ans = decode_to_string(decoded_ids, formula_vocab)

                temp_data = x.copy()
                temp_data["predicted"] = ans.replace("<sos>","").replace("<eos>","")
                new_data.append(temp_data)
    elif args.model_type in ['lstm_lstm', 'lstm_lstm_attn']:
        with torch.no_grad():
            for x in data:
                question = x["Problem"]
                input_tensor = preprocess_problem(question, special_embeddings, glove_vector).unsqueeze(0).to(device)

                encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)
                _, decoded_ids, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_cell, None)

                # _, topi = decoder_outputs.topk(1)
                # decoded_ids = topi.squeeze()

                ans = decode_to_string(decoded_ids, formula_vocab)

                temp_data = x.copy()
                temp_data["predicted"] = ans.replace("<sos>","").replace("<eos>","")
                new_data.append(temp_data)

    # Writing the predictions to output file
    with open(file_path, 'w') as file:
        json.dump(new_data, file, indent=4)

    print("Output written to", file_path)

    # print('Model file:', args.model_file)
    # print('Beam size:', args.beam_size)
    # print('Model type:', args.model_type)
    # print('Test data file:', args.test_data_file)

if __name__ == "__main__":
    main()
