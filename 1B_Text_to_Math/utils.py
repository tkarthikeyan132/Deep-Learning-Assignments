import re
from collections import Counter, OrderedDict
import torchtext
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os

# Function to remove commas from matched integers
def remove_commas(match):
    return match.group(0).replace(',', '')

#get n0, n1, .. from problem statement
def get_nums(sent):

    #match the digits
    pt = r'[-+]?\d*\.\d+|\d+'

    #sent = re.sub(sent, remove_commas, sent)
    matches = re.findall(pt, sent)
    
    numbers = [match for match in matches]
    return numbers

#Tokenize the linear formula
def tokenize_LF(linear_formula):
    operations = linear_formula.split("|")
    glist = []
    
    if "" in operations:
        operations.remove("")
    
    for op in operations:
        lst1 = op.split("(")

        if ")" in lst1[1]:
            lst1[1] = lst1[1].replace(")","")
        
        glist.append(lst1[0])   

        params = lst1[1].split(",")
        params_with_comma = []
        for par in params:
            params_with_comma.append(par)
            params_with_comma.append(",")
        
        glist.extend(["("] + params_with_comma[:-1] + [")", "|"])

    if linear_formula[-1] != "|":
        glist = glist[:-1]
    
    return glist

#Remove comma from the pattern (1,000,000)
def remove_num_comma(question):
    pattern = r'\b\d{1,3}(,\d{3})+\b'
    ##### handles 1,000,000 and convert it into 1000000 #####
    question = re.sub(pattern, remove_commas, question)
    return question

#Replace actual values with variables in the question
def get_mask_question(question):
    question = remove_num_comma(question)
    nums = get_nums(question)
    replacements = dict()
    
    for i,num in enumerate(nums):
        pattern = rf'\b{num}\b'
        compiled_pattern = re.compile(pattern)
        # Perform the replacement
        question = compiled_pattern.sub(f"n{i}", question, count=1)

    return question

#Return the problem vocab and linear formula vocab based on the data
def build_input_vocab(data):
    problem_tokens_list = []
    formula_tokens_list = []

    for entry in data:
        masked_question = get_mask_question(entry["Problem"])
        question_toks = masked_question.split()
        
        LF_toks = tokenize_LF(entry["linear_formula"])
        
        problem_tokens_list += question_toks
        formula_tokens_list += LF_toks
    
    problem_counter = Counter(problem_tokens_list)
    formula_counter = Counter(formula_tokens_list)
    
    sorted_by_freq = sorted(problem_counter.items(), key=lambda x: x[1], reverse=True)
    problem_ordered_dict = OrderedDict(sorted_by_freq)

    sorted_by_freq = sorted(formula_counter.items(), key=lambda x: x[1], reverse=True)
    formula_ordered_dict = OrderedDict(sorted_by_freq)
    
    print("Length of problem vocab built is ", len(problem_ordered_dict))
    print("Length of formula vocab built is ", len(formula_ordered_dict))
    
    problem_vocab = torchtext.vocab.vocab(problem_ordered_dict, specials=['<pad>', '<sos>', '<eos>', '<unk>'])
    problem_vocab.set_default_index(problem_vocab['<unk>'])

    formula_vocab = torchtext.vocab.vocab(formula_ordered_dict, specials=['<pad>', '<sos>', '<eos>', '<unk>'])
    formula_vocab.set_default_index(formula_vocab['<unk>'])
    
    return problem_vocab, formula_vocab

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

#Convert the decoded_ids to string
def decode_to_string(decoded_ids, formula_vocab):
    string = ""
    for x in list(decoded_ids):
        if x != 0:
            string += formula_vocab.get_itos()[x] 
    return string

def plot_loss_graph(train_loss_list, val_loss_list, epochs, output_path):
    #Loss vs epoch graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig(output_path + 'loss_plot.png')

def store_json_file(train_loss_list, val_loss_list, HIDDEN_DIM, BATCH_SIZE, EMBED_DIM, LEARNING_RATE, EPOCHS, TEACHER_FORCING, output_dir):
    result = {
        "train loss": train_loss_list[-1],
        "val loss": val_loss_list[-1],
        "Hyperparameters":{
            "HIDDEN SIZE": HIDDEN_DIM,
            "BATCH SIZE": BATCH_SIZE,
            "EMBED DIM": EMBED_DIM,
            "LEARNING RATE": LEARNING_RATE,
            "TEACHER_FORCING": TEACHER_FORCING,
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