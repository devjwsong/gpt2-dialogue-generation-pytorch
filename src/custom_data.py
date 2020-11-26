from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain

import torch
import copy


class CustomDataset(Dataset):
    def __init__(self, data_type, config):
        assert data_type == 'train' or data_type == 'valid', "Data type incorrect. It should be 'train' or 'valid'."
        
        if data_type == 'train':
            data_name = config['train_name']
        elif data_type == 'valid':
            data_name = config['valid_name']
        
        print(f"Loading {data_name}_id.txt...")
        with open(f"{config['data_dir']}/{data_name}.id", 'r') as f:
            lines = f.readlines()
        
        self.input_ids = []  # (N, L)
        self.attention_masks = []  # (N, L)
        self.token_type_ids = []  # (N, L)
        self.labels = []  # (N, L)
        
        print(f"Processing {data_name}.id...")
        dialogues = []
        dialogue = []
        cur_speaker = 1
        for i, line in enumerate(tqdm(lines)):
            if line.strip() == config['dialogue_split_line']:
                dialogue = []
                cur_speaker = 1
            else:
                if cur_speaker == 1:
                    speaker_id = config['speaker1_id']
                else:
                    speaker_id = config['speaker2_id']
                
                token_ids = line.strip().split(' ')
                token_ids = [speaker_id] + [int(idx) for idx in token_ids]                    
                
                if len(dialogue) < config['max_time']:
                    dialogue.append(token_ids)
                else:
                    dialogue = dialogue[1:] + [token_ids]
                    
                cur_speaker = (cur_speaker % 2) + 1
                dialogues.append(copy.deepcopy(dialogue))
        
        for d, dialogue in enumerate(tqdm(dialogues)):
            if len(dialogue) > 1:
                dialogue[0] = [config['bos_id']] + dialogue[0]
                dialogue[-1] = dialogue[-1] + [config['eos_id']]
                
                total_len = 0
                for utter in dialogue:
                    total_len += len(utter)
                    
                if total_len > config['max_len']:
                    should_cut = True
                else:
                    should_cut = False
                    
                if should_cut:
                    dialogue = [utter[:config['utter_len']] for utter in dialogue]
                    dialogue[-1][-1] = config['eos_id']
                    
                token_type_id = [[utter[0]] * len(utter) if u != 0 else [utter[1]] * len(utter) for u, utter in enumerate(dialogue)]
                lm_label = [[-100] * len(utter) if u != len(dialogue)-1 else utter for u, utter in enumerate(dialogue)]
                input_id = list(chain.from_iterable(dialogue))
                token_type_id = list(chain.from_iterable(token_type_id))
                lm_label = list(chain.from_iterable(lm_label))
                
                assert len(input_id) == len(lm_label) and len(input_id) == len(token_type_id), "There is something wrong in dialogue process."
                
                input_id, token_type_id, lm_label, attention_mask = self.make_padding(input_id, token_type_id, lm_label, config['max_len'], config['pad_id'])
                
                self.input_ids.append(input_id)
                self.attention_masks.append(attention_mask)
                self.token_type_ids.append(token_type_id)
                self.labels.append(lm_label)
                
        self.input_ids = torch.LongTensor(self.input_ids)  # (N, L)
        self.attention_masks = torch.FloatTensor(self.attention_masks)  # (N, L)
        self.token_type_ids = torch.LongTensor(self.token_type_ids)  # (N, L)
        self.labels = torch.LongTensor(self.labels)  # (N, L)
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.token_type_ids[idx], self.labels[idx]
    
    def make_padding(self, input_id, token_type_id, lm_label, max_len, pad_id):
        left = max_len - len(input_id)

        attention_mask = [1] * len(input_id) + [0] * left
        input_id += [pad_id] * left
        token_type_id += [pad_id] * left
        lm_label += [-100] * left

        return input_id, token_type_id, lm_label, attention_mask