from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain

import torch
import copy
import json


class CustomDataset(Dataset):
    def __init__(self, prefix, args):
        assert prefix == args.train_prefix or prefix == args.valid_prefix
        
        print(f"Loading {prefix}_id.json...")
        with open(f"{args.data_dir}/{prefix}_ids.json", 'r') as f:
            dials = json.load(f)
        
        self.input_ids = []  # (N, L)
        self.token_type_ids = []  # (N, L)
        self.labels = []  # (N, L)
        
        print(f"Processing {prefix} data...")
        for dial in tqdm(dials):
            hists = []
            for u, utter in enumerate(dial):
                if u % 2 == 0:
                    hists.append([args.sp1_id] + utter)
                else:
                    hists.append([args.sp2_id] + utter)
                    
            for h in range(len(hists)):
                if hists[h][0] == args.sp2_id:
                    start = max(0, h-args.max_turns+1)
                    for s in range(start, h):
                        contexts = hists[s:h+1]
                        input_ids = [args.bos_id] + list(chain.from_iterable(contexts)) + [args.eos_id]
                        if len(input_ids) <= args.max_len:
                            start_sp_id, next_sp_id = contexts[0][0], contexts[1][0]
                            token_type_ids = [[start_sp_id] * len(ctx) if c % 2 == 0 else [next_sp_id] * len(ctx) for c, ctx in enumerate(contexts)]
                            assert token_type_ids[-1][0] == args.sp2_id
                            token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [args.sp2_id]
                            assert len(input_ids) == len(token_type_ids)
                            
                            labels = [[-100] * len(ctx) if c < len(contexts)-1 else [-100] + ctx[1:] for c, ctx in enumerate(contexts)]
                            assert labels[-1][1:] == contexts[-1][1:]
                            labels = [-100] + list(chain.from_iterable(labels)) + [args.eos_id]
                            assert len(input_ids) == len(labels)
                            
                            self.input_ids.append(input_ids)
                            self.token_type_ids.append(token_type_ids)
                            self.labels.append(labels)
                            
                            break
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx]
    
    
class PadCollate():
    def __init__(self, eos_id):
        self.eos_id = eos_id
        
    def pad_collate(self, batch):
        input_ids, token_type_ids, labels =[], [], []
        for idx, seqs in enumerate(batch):
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[1]))
            labels.append(torch.LongTensor(seqs[2]))
            
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=self.eos_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
        return input_ids, token_type_ids, labels
