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
        with open(f"{config['data_dir']}/{config['matched_dir']}/{data_name}.id", 'r') as f:
            lines = f.readlines()
        
        self.input_ids = []  # (N, C, L)
        self.attention_masks = []  # (N, C, L)
        self.token_type_ids = []  # (N, C, L)
        self.mc_token_ids = []  # (N, C)
        self.lm_labels = []  # (N, C, L)
        self.mc_labels = []  # (N)
        
        print(f"Processing {data_name}_id.txt...")
        input_group = []
        mask_group = []
        token_type_group = []
        mc_token_group = []
        lm_labels_group = []
        for l, line in enumerate(tqdm(lines)):
            comps = line.strip().split('\t')
            start_speaker = int(comps[0])
            histories = comps[1].split(config['utter_split_symbol'])
            target = comps[2]
            label = int(comps[3])
            
            history_ids = [[int(idx) for idx in history.split(' ')] for history in histories]
            target_ids = [int(idx) for idx in target.split()]
            
            input_id, token_type_id, lm_label = \
                self.make_seqs(
                    history_ids, target_ids, config['bos_id'], config['eos_id'], 
                    start_speaker, config['speaker1_id'], config['speaker2_id'], 
                    config['max_len'], label=label
            )  # (L), (L), (L)   
            
            assert len(input_id) == len(token_type_id) and len(token_type_id) == len(lm_label)
            
            input_id, token_type_id, lm_label, attention_mask, mc_token_id =\
                self.make_padding(input_id, token_type_id, lm_label, config['max_len'], config['pad_id'])
            
            input_group.append(input_id)
            mask_group.append(attention_mask)
            token_type_group.append(token_type_id)
            mc_token_group.append(mc_token_id)
            lm_labels_group.append(lm_label)
            
            if label == 0:
                self.input_ids.append(input_group)
                self.attention_masks.append(mask_group)
                self.token_type_ids.append(token_type_group)
                self.mc_token_ids.append(mc_token_group)
                self.lm_labels.append(lm_labels_group)
                self.mc_labels.append(config['num_distractors'])
                
                input_group = []
                mask_group = []
                token_type_group = []
                mc_token_group = []
                lm_labels_group = []
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.token_type_ids[idx],
            self.mc_token_ids[idx], self.lm_labels[idx], self.mc_labels[idx]

    def make_seqs(history_ids, target_ids, bos_id, eos_id, start_speaker, speaker1_id, speaker2_id, max_len, label=0):
        input_id = history_ids + [target_ids]
        cur_speaker = copy.deepcopy(start_speaker)
        total_len = 0
        for i, token_ids in enumerate(input_id):
            if cur_speaker == 1:
                speaker_id = copy.deepcopy(speaker1_id)
            elif cur_speaker == 2:
                speaker_id = copy.deepcopy(speaker2_id)
                
            new_token_ids = [speaker_id] + token_ids
            cur_speaker = (cur_speaker % 2) + 1
            
            if i == 0:
                new_token_ids = [bos_id] + new_token_ids
            elif i == len(input_id)-1:
                new_token_ids = new_token_ids + [eos_id]
                
            input_id[i] = new_token_ids
            total_len += len(new_token_ids)
        
        if total_len > max_len:
            remove_idx = 0
            remove_len = len(input_id[remove_idx])
            while True:
                if (total_len - remove_len) <= max_len-1:
                    break
                else:
                    remove_idx += 1
                    remove_len += len(input_id[remove_idx])
                    
            input_id = input_id[remove_idx+1:]
            input_id[0] = [bos_id] + input_id[0]
            
        token_type_id = []
        for i, utter in enumerate(input_id):
            if i == 0:
                speaker_id = utter[1]
            else:
                speaker_id = utter[0]
                
            token_type_id.append([speaker_id] * len(utter))
        
        if label == 0:  # golden reply
            lm_label = [[-100] * len(utter) if i != len(input_id)-1 else utter for i, utter in enumerate(input_id)]
        else:  # distractor
            lm_label = [[-100] * len(utter) for utter in input_id]
            
        input_id = list(chain.from_iterable(input_id))
        token_type_id = list(chain.from_iterable(token_type_id))
        lm_label = list(chain.from_iterable(lm_label))
        
        return input_id, token_type_id, lm_label
    
    def make_padding(input_id, token_type_id, lm_label, max_len, pad_id):
        mc_token_id = len(input_id)-1
        left = max_len - len(input_id)

        attention_mask = [1] * len(input_id) + [0] * left
        input_id += [pad_id] * left
        token_type_id += [pad_id] * left
        lm_label += [-100] * left

        return input_id, token_type_id, lm_label, attention_mask, mc_token_id