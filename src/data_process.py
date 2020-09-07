from tqdm import tqdm
from transformers import *

import torch
import os


# Parameters for data
data_dir = 'data'
raw_data_dir = 'raw'
sp_dir = 'trained_sp'
sp_prefix = 'sp'
processed_dir = 'processed'
train_name = 'train'
valid_name = 'validation'
test_name = 'test'
raw_name_prefix = 'dialogues'
train_frac = 0.8
end_of_utterance = '__eou__'
space = 'Ġ'
end_marks = ['.', ',', '?', '!', '...']
quotes = ['"', '\'', '’']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']
dialogue_split_line = "[END OF DIALOGUE]"

special_tokens_dict = {
    'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token': '<unk>',
    'additional_special_tokens': ['<s1>', '<s2>']
}


def merge_data(total_lines, data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
    total_lines += lines
    
    return total_lines


def resplit_data(total_lines):
    train_lines = total_lines[:int(len(total_lines) * train_frac)]
    valid_lines = total_lines[int(len(total_lines) * train_frac):]
    
    return train_lines, valid_lines


def process_token_list(token_list):
    for i, token in enumerate(token_list):
        if space in token:
            if token[1:] in end_marks or token[1:] in abbreviations:
                token_list[i] = token[1:]
                
        if token in end_marks:
            if i<len(token_list)-1 and space not in token_list[i+1]:
                token_list[i+1] = space + token_list[i+1]
    
    quote_count = 0
    for i, token in enumerate(token_list):
        if space in token:
            if token[1:] in quotes:
                if token[1:] == quotes[1] and i<len(token_list)-1 and token_list[i+1] in abbreviations:
                    token_list[i] = token[1:]
                else:
                    if quote_count % 2 == 1:
                        token_list[i] = token[1:]
                    else:
                        if i<len(token_list)-1 and token_list[i+1][0]==space:
                            token_list[i+1] = token_list[i+1][1:]
                    quote_count += 1
                
    new_token_list = [token for token in token_list if token != space]
        
    return new_token_list


def save_data(lines, tokenizer, name):
    texts = []
    ids = []
    for line in tqdm(lines):
        dialogue = line.strip().replace(' __eou__ ', '__eou__')
        dialogue = dialogue.replace(' __eou__', '__eou__')
        dialogue = dialogue.replace('__eou__ ', '__eou__')

        utters = dialogue.split('__eou__')[:-1]
        dialogue_ids = []
        
        for utter in utters:
            token_list = tokenizer.tokenize(utter.replace(quotes[2], quotes[1]))
            token_list = process_token_list(token_list)
            
            text = tokenizer.convert_tokens_to_string(token_list)
            texts.append(text)
            
            token_ids = tokenizer(text)['input_ids']
            
            dialogue_ids.append(token_ids)
        
        texts.append(dialogue_split_line)
        ids.append(dialogue_ids)
    
    print(f"Saving {name} text file...")
    with open(f"{data_dir}/{processed_dir}/{name}.txt", 'w') as f:
        for text in tqdm(texts):
            f.write(f"{text}\n")
    
    print(f"Saving {name} id file...")
    with open(f"{data_dir}/{processed_dir}/{name}_id.txt", 'w') as f:
        for dialogue in tqdm(ids):
            for utter in dialogue:
                utter_str = [str(idx) for idx in utter]
                f.write(f"{' '.join(utter_str)}\n")
            f.write(f"{dialogue_split_line}\n")


if __name__=='__main__':
    print("Merging all dialogue dataset...")
    total_lines = merge_data([], f"{data_dir}/{raw_data_dir}/dialogues_{train_name}.txt")
    total_lines = merge_data(total_lines, f"{data_dir}/{raw_data_dir}/dialogues_{valid_name}.txt")
    total_lines = merge_data(total_lines, f"{data_dir}/{raw_data_dir}/dialogues_{test_name}.txt")
    
    print("Respliting data...")
    train_lines, valid_lines = resplit_data(total_lines)
    
    print("Loading GPT2 Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(special_tokens_dict)
    
    if not os.path.isdir(f"{data_dir}/{processed_dir}"):
        os.mkdir(f"{data_dir}/{processed_dir}")
    
    print("Processing train utterances...")
    save_data(train_lines, tokenizer, train_name)
    print("Processing valid utterances...")
    save_data(valid_lines, tokenizer, valid_name)            
    
    print("Data preprocess finished!")
