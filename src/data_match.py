from tqdm import tqdm
from transformers import *

import torch
import argparse
import os
import random
import time
import json


def make_dialogue_list(unmatched_dir, name, dialogue_split_line):
    print(f"Loading dialogues from {name} set...")
    with open(f"{unmatched_dir}/{name}.txt", 'r') as f:
        lines = f.readlines()
        
    dialogues = []
    dialogue = []
    for line in tqdm(lines):
        if line.strip() == dialogue_split_line:
            dialogues.append(dialogue)
            dialogue = []
        else:
            dialogue.append(line.strip())
            
    return dialogues


def process_dialogue_list(dialogues, other_dialogues, utter_split_symbol, num_distractors, max_times, tokenizer):
    text_lines = []
    id_lines = []
    start_speakers = []
    random.seed(int(time.time()))
    
    for dialogue in tqdm(dialogues):
        text_histories = []
        id_histories = []
        start_speaker = 1
        for i, utter in enumerate(dialogue):
            if i < len(dialogue)-1:
                if len(text_histories) == max_times:
                    text_histories = text_histories[1:]
                    id_histories = id_histories[1:]
                    start_speaker = (start_speaker % 2) + 1
                
                text_histories.append(utter)
                text_str = utter_split_symbol.join(text_histories)
                
                utter_ids = tokenizer(utter)['input_ids']
                utter_ids = [str(token) for token in utter_ids]
                utter_ids = ' '.join(utter_ids)
                id_histories.append(utter_ids)
                id_str = utter_split_symbol.join(id_histories)
                
                for d in range(num_distractors):
                    distractor = get_distractor(other_dialogues)
                    text_lines.append(f"{text_str}\t{distractor}\t1")
                    
                    distractor_ids = tokenizer(distractor)['input_ids']
                    distractor_ids = [str(token) for token in distractor_ids]
                    distractor_ids = ' '.join(distractor_ids)
                    id_lines.append(f"{id_str}\t{distractor_ids}\t1")
                    
                    start_speakers.append(start_speaker)
                
                reply = dialogue[i+1]
                text_lines.append(f"{text_str}\t{reply}\t0")
                
                reply_ids = tokenizer(reply)['input_ids']
                reply_ids = [str(token) for token in reply_ids]
                reply_ids = ' '.join(reply_ids)
                id_lines.append(f"{id_str}\t{reply_ids}\t0")
                
                start_speakers.append(start_speaker)
                    
    return text_lines, id_lines, start_speakers
                    

def get_distractor(other_dialogues):
    dialogue_idx = random.randint(0, len(other_dialogues)-1)
    samples = other_dialogues[dialogue_idx]

    utter_idx = random.randint(0, len(samples)-1)
    distractor = samples[utter_idx]
    
    return distractor


def save_file(text_lines, id_lines, start_speakers, matched_dir, name):
    print(f"Saving files for {name} set...")
    with open(f"{matched_dir}/{name}.txt", 'w') as f:
        for i, line in enumerate(tqdm(text_lines)):
            f.write(f"{start_speakers[i]}\t{line}\n")
            
    with open(f"{matched_dir}/{name}.id", 'w') as f:
        for line in tqdm(id_lines):
            f.write(f"{start_speakers[i]}\t{line}\n")
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help="The path to configuration file.")
    
    args = parser.parse_args()
    
    print("Loading configurations...")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        
    print("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'bos_token': config['bos'],
        'eos_token': config['eos'],
        'pad_token': config['pad'],
        'unk_token': config['unk'],
        'additional_special_tokens': [config['speaker1'], config['speaker2']]
    }
    tokenizer.add_special_tokens(special_tokens)
        
    unmatched_dir = f"{config['data_dir']}/{config['unmatched_dir']}"
    
    train_dialogues = make_dialogue_list(unmatched_dir, config['train_name'], config['dialogue_split_line'])
    valid_dialogues = make_dialogue_list(unmatched_dir, config['valid_name'], config['dialogue_split_line'])
    
    print("Processing / Distractor matching...")
    train_text_lines, train_id_lines, train_start_speakers =\
        process_dialogue_list(train_dialogues, valid_dialogues, config['utter_split_symbol'], config['num_distractors'], config['max_time'], tokenizer)
    valid_text_lines, valid_id_lines, valid_start_speakers =\
        process_dialogue_list(valid_dialogues, train_dialogues, config['utter_split_symbol'], config['num_distractors'], config['max_time'], tokenizer)
    
    matched_dir = f"{config['data_dir']}/{config['matched_dir']}"
    if not os.path.isdir(matched_dir):
        os.mkdir(matched_dir)
    
    save_file(train_text_lines, train_id_lines, train_start_speakers, matched_dir, config['train_name'])
    save_file(valid_text_lines, valid_id_lines, valid_start_speakers, matched_dir, config['valid_name'])
    
    print("Ready to train!")
