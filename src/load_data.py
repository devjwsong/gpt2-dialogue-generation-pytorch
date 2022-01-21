from tqdm import tqdm
from transformers import GPT2Tokenizer
from process_data import *

import argparse
import os
import json


dataset_list = ['daily_dialog', 'empathetic_dialogues', 'persona_chat', 'blended_skill_talk']


def merge_data(tokenizer, args):
    train_dialogues = []
    valid_dialogues = []
    num_train = 0
    num_valid = 0
    for data_name in dataset_list:
        print(f"Processing {data_name}...")
        if data_name == 'daily_dialog':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_daily(tokenizer, args.train_frac)
        elif data_name == 'empathetic_dialogues':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_empathetic(tokenizer, args.train_frac)
        elif data_name == 'persona_chat':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_persona(tokenizer, args.train_frac)
        elif data_name == 'blended_skill_talk':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_blended(tokenizer, args.train_frac)
        
        train_dialogues += part_train_dialogues
        valid_dialogues += part_valid_dialogues
    
        print("#"*50 + f"Analysis on {data_name}" + "#"*50)
        print(f"The number of train dialogues: {len(part_train_dialogues)}")
        print(f"The number of valid dialogues: {len(part_valid_dialogues)}")    
        print(f"The number of train utterances: {part_num_train}")    
        print(f"The number of valid utterances: {part_num_valid}")
        
        num_train += part_num_train
        num_valid += part_num_valid
        
    return train_dialogues, valid_dialogues, num_train, num_valid


def save_data(prefix, data_dir, dialogues, tokenizer):
    print(f"Saving {prefix} text file...")
    with open(f"{data_dir}/{prefix}_utters.json", 'w') as f:
        json.dump(dialogues, f)
    
    print(f"Saving {prefix} idx file...")
    ids = []
    for dialogue in tqdm(dialogues):
        dialogue_ids = []
        for utter in dialogue:
            tokens = tokenizer.tokenize(utter)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            dialogue_ids.append(token_ids)
        ids.append(dialogue_ids)
        
    assert len(ids) == len(dialogues)
        
    with open(f"{data_dir}/{prefix}_ids.json", 'w') as f:
        json.dump(ids, f)

        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument('--train_frac', type=float, default=0.85, help="The ratio of the conversations to be included in the train set.")
    parser.add_argument('--model_type', type=str, default="gpt2", help="The model type of GPT-2.")
    
    args = parser.parse_args()
    
    assert args.model_type in [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"
    ]
    
    print("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    
    args.data_dir = f"{args.data_dir}/{args.model_type}"
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    
    print("Loading & Merging all datasets...")
    train_dialogues, valid_dialogues, num_train, num_valid = merge_data(tokenizer, args)
    
    print("Saving train data...")
    save_data(args.train_prefix, args.data_dir, train_dialogues, tokenizer)
    print("Saving validation data...")
    save_data(args.valid_prefix, args.data_dir, valid_dialogues, tokenizer)            

    print("#"*50 + "Analysis on total data" + "#"*50)
    print(f"The number of train dialogues: {len(train_dialogues)}")
    print(f"The number of valid dialogues: {len(valid_dialogues)}")    
    print(f"The number of train utterances: {num_train}")    
    print(f"The number of valid utterances: {num_valid}")
    
