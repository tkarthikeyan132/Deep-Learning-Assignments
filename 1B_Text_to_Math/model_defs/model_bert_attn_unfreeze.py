import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import BertModel, BertTokenizer

#PARAMETER
MAX_LENGTH = 96

#Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = False
        # Update the last layer to be trainable
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_state = outputs.last_hidden_state[:,0,:].unsqueeze(dim=0)
        cell_state = outputs.pooler_output.unsqueeze(dim=0)
        return hidden_states, hidden_state, cell_state
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, teacher_forcing_ratio = 1.0, beam_width=1):
        super(DecoderRNN, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_width = beam_width

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, encoder_cell, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(1)
        decoder_hidden = encoder_hidden.contiguous()
        decoder_cell = encoder_cell.contiguous()
        decoder_outputs = []
        attentions = []

        beam = [([1], (decoder_hidden, decoder_cell), 0)]

        if self.training:
            max_length = target_tensor.shape[1]
        else:
            max_length = MAX_LENGTH

        for i in range(max_length):
            

            #During Training
            if target_tensor is not None:
                decoder_output, decoder_hidden, decoder_cell, attn_weights = self.forward_step(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                )
                decoder_outputs.append(decoder_output)
                attentions.append(attn_weights)
                # Teacher forcing: Feed the target as the next input
                if random.random() < self.teacher_forcing_ratio:
                    decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()  # detach from history as input   
                    
            else: #During Inference
                # Without teacher forcing: use its own predictions as the next input
                # _, topi = decoder_output.topk(1)
                # decoder_input = topi.squeeze(-1).detach()  # detach from history as input
                new_beam = []
                for sequence, (hidden, cell), score in beam:
                    input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(sequence[-1])
                    decoder_output, decoder_hidden, decoder_cell, _ = self.forward_step(input, hidden, cell, encoder_outputs)
                    
                    top_vals, top_inds = decoder_output.squeeze(1).topk(self.beam_width, dim=1)

                    for j in range(self.beam_width):
                        new_word_idx = top_inds[0][j]
                        new_seq = sequence + [new_word_idx.item()]
                        updated_score = score - torch.log(top_vals[0][j])
                        new_beam.append((new_seq, (decoder_hidden, decoder_cell), updated_score))

                new_beam.sort(key=lambda x: x[2])
                beam = new_beam[:self.beam_width]

        if target_tensor is not None:
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            attentions = torch.cat(attentions, dim=1)
        else:
            decoded_ids = beam[0][0]
            attentions = None

        return decoder_outputs, decoded_ids, decoder_hidden, attentions


    def forward_step(self, input, hidden, cell, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(input_lstm, (hidden, cell))
        output = self.out(output)

        return output, hidden, cell, attn_weights