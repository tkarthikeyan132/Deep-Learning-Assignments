import torch
from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ProblemFormulaDataset(Dataset):
    def __init__(self, data, problem_vocab, glove_vector, special_embeddings, formula_vocab):
        self.data = data

        self.problem_vocab = problem_vocab
        self.glove_vector = glove_vector
        self.special_embeddings = special_embeddings
        self.formula_vocab = formula_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        problem = self.preprocess_problem(sample["Problem"])
        linear_formula = self.preprocess_formula(sample["linear_formula"])
        
        return problem, linear_formula

    def preprocess_problem(self, question):
        masked_question = get_mask_question(question)
        question_toks = masked_question.split()
    
        embeddings = []
        embeddings.append(torch.tensor(self.special_embeddings['<sos>'], dtype=torch.float32))
        for word in question_toks:
            if word in self.glove_vector.stoi:
                embeddings.append(torch.tensor(self.glove_vector[word], dtype=torch.float32))
            else:
                embeddings.append(torch.tensor(self.special_embeddings['<unk>'], dtype=torch.float32))
        
        embeddings.append(torch.tensor(self.special_embeddings['<eos>'], dtype=torch.float32))
        
        return torch.stack(embeddings)
        
    def preprocess_formula(self, formula):
        LF_toks = tokenize_LF(formula)
        LF_toks = ['<sos>'] + LF_toks + ['<eos>']
    
        LF_embeddings = [self.formula_vocab[word] for word in LF_toks]
        
        return torch.tensor(LF_embeddings)

def collate_fn(batch):
    problems, linear_formulas = zip(*batch)
    
    # Pad sequences
    problems_padded = pad_sequence(problems, batch_first=True, padding_value=0)
    linear_formulas_padded = pad_sequence(linear_formulas, batch_first=True, padding_value=0)

    return problems_padded, linear_formulas_padded


####### BERT MODEL #######

class ProblemFormulaDatasetBert(Dataset):
    def __init__(self, data, formula_vocab, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formula_vocab = formula_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_ids, attention_mask = self.preprocess_problem(sample["Problem"])
        linear_formula = self.preprocess_formula(sample["linear_formula"])
        
        return input_ids, attention_mask, linear_formula

    def preprocess_problem(self, question):
        masked_question = get_mask_question(question)
        encoding = self.tokenizer(masked_question, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask
        
    def preprocess_formula(self, formula):
        LF_toks = tokenize_LF(formula)
        LF_toks = ['<sos>'] + LF_toks + ['<eos>']
    
        LF_embeddings = [self.formula_vocab[word] for word in LF_toks]
        
        return torch.tensor(LF_embeddings)
    
def collate_fn_bert(batch):
    problems, attention_masks, linear_formulas = zip(*batch)
    
    # Pad sequences
    problems_padded = pad_sequence(problems, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    linear_formulas_padded = pad_sequence(linear_formulas, batch_first=True, padding_value=0)

    return problems_padded, attention_masks_padded, linear_formulas_padded