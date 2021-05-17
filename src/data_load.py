from tqdm import tqdm
from transformers import *
from data_process import *

import argparse
import os
import json


dataset_list = ['persona_chat']


def merge_data(tokenizer, train_frac):
    train_dialogues = []
    valid_dialogues = []
    total_train_utter_num = 0
    total_valid_utter_num = 0
    for data_name in dataset_list:
        print(f"Processing {data_name}...")
        if data_name == 'daily_dialog':
            partial_train_dialogues, partial_valid_dialogues, train_utter_num, valid_utter_num = load_daily_dialog(tokenizer, train_frac)
        elif data_name == 'empathetic_dialogues':
            partial_train_dialogues, partial_valid_dialogues, train_utter_num, valid_utter_num = load_empathetic_dialogues(tokenizer, train_frac)
        elif data_name == 'persona_chat':
            partial_train_dialogues, partial_valid_dialogues, train_utter_num, valid_utter_num = load_persona_chat(tokenizer, train_frac)
        elif data_name == 'blended_skill_talk':
            partial_train_dialogues, partial_valid_dialogues, train_utter_num, valid_utter_num = load_blended_skill_talk(tokenizer, train_frac)
        
        train_dialogues += partial_train_dialogues
        valid_dialogues += partial_valid_dialogues
    
        print(f"#################### Analysis on {data_name} ####################")
        print(f"The number of train dialogues: {len(partial_train_dialogues)}")
        print(f"The number of valid dialogues: {len(partial_valid_dialogues)}")    
        print(f"The number of train utterances: {train_utter_num}")    
        print(f"The number of valid utterances: {valid_utter_num}")
        
        total_train_utter_num += train_utter_num
        total_valid_utter_num += valid_utter_num
        
    return train_dialogues, valid_dialogues, total_train_utter_num, total_valid_utter_num


def save_data(dialogues, name, dialogue_split_line, data_dir):
    print(f"Saving {name} text file...")
    with open(f"{data_dir}/{name}.txt", 'w') as f:
        for dialogue in tqdm(dialogues):
            for utter in dialogue:
                f.write(f"{utter}\n")
            f.write(f"{dialogue_split_line}\n")
     
    print(f"Saving {name} idx file...")
    with open(f"{data_dir}/{name}.id", 'w') as f:
        for dialogue in tqdm(dialogues):
            for utter in dialogue:
                token_ids = tokenizer(utter)['input_ids']
                token_ids = ' '.join([str(idx) for idx in token_ids])
                f.write(f"{token_ids}\n")
            f.write(f"{dialogue_split_line}\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help="The path to configuration file.")
    
    args = parser.parse_args()
    
    print("Loading configurations...")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print("Loading & Merging all datasets...")
    train_dialogues, valid_dialogues, total_train_utter_num, total_valid_utter_num = merge_data(tokenizer, config['train_frac'])
    
    if not os.path.isdir(config['data_dir']):
        os.mkdir(config['data_dir'])
    
    print("Saving train data...")
    save_data(train_dialogues, config['train_name'], config['dialogue_split_line'], config['data_dir'])
    print("Saving validation data...")
    save_data(valid_dialogues, config['valid_name'], config['dialogue_split_line'], config['data_dir'])            
    
    print("Data preprocess finished!")

    print(f"#################### Analysis on total data ####################")
    print(f"The number of train dialogues: {len(train_dialogues)}")
    print(f"The number of valid dialogues: {len(valid_dialogues)}")    
    print(f"The number of train utterances: {total_train_utter_num}")    
    print(f"The number of valid utterances: {total_valid_utter_num}")
    
