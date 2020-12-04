from transformers import *
from custom_data import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from itertools import chain

import torch
import os, sys
import numpy as np
import argparse
import time
import copy
import json


class Manager():
    def __init__(self, config_path, mode, ckpt_name=None):
        print("Setting the configurations...")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        if self.config['device'] == "cuda":
            self.config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif self.config['device'] == "cpu":
            self.config['device'] = torch.device('cpu')
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {
            'bos_token': self.config['bos'],
            'eos_token': self.config['eos'],
            'pad_token': self.config['pad'],
            'additional_special_tokens': [self.config['speaker1'], self.config['speaker2']]
        }
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.config['vocab_size'] = len(vocab)
        self.config['bos_id'] = vocab[self.config['bos']]
        self.config['eos_id'] = vocab[self.config['eos']]
        self.config['pad_id'] = vocab[self.config['pad']]
        self.config['speaker1_id'] = vocab[self.config['speaker1']]
        self.config['speaker2_id'] = vocab[self.config['speaker2']]
        
        self.config['utter_len'] = (self.config['max_len']-self.config['max_time']-2) // self.config['max_time']
        
        # Load model    
        print("Loading the model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.config['device'])
        self.model.resize_token_embeddings(self.config['vocab_size'])
            
        if mode == 'train':            
            # Load optimizer
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
            self.best_loss = sys.float_info.max
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset('train', self.config)
            valid_set = CustomDataset('valid', self.config)
            self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config['batch_size'])
            self.valid_loader = DataLoader(valid_set, shuffle=True, batch_size=self.config['batch_size'])
            
        if not os.path.exists(self.config['ckpt_dir']):
            os.mkdir(self.config['ckpt_dir'])
        
        if ckpt_name is not None:
            assert os.path.exists(f"{self.config['ckpt_dir']}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if args.mode == 'train':
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.best_loss = checkpoint['loss']
              
        print("Setting finished.")
              
    def train(self):
        print("Training starts.")
              
        for epoch in range(1, self.config['num_epochs']+1):
            self.model.train()
            
            print(f"#################### Epoch: {epoch} ####################")
            train_losses = []
            train_ppls = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, lm_labels = batch
                input_ids, token_type_ids, lm_labels = \
                    input_ids.to(self.config['device']), token_type_ids.to(self.config['device']), lm_labels.to(self.config['device'])
                
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = lm_labels
                )
                
                loss, logits = outputs[0], outputs[1]
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                train_losses.append(loss.item())
                train_ppls.append(torch.exp(loss).item())
            
            train_loss = np.mean(train_losses)
            train_ppl = np.mean(train_ppls)
            print(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")
            
            valid_loss, valid_ppl = self.validation()
              
            if valid_loss < self.best_loss:
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': self.best_loss,
                }
              
                torch.save(state_dict, f"{self.config['ckpt_dir']}/best_ckpt.tar")
                print(f"***** Current best checkpoint is saved. *****")
                self.best_loss = valid_loss
              
            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")
              
        print("Training finished!")
    
    def validation(self):
        print("Validation processing...")
        self.model.eval()
              
        valid_losses = []
        valid_ppls = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, lm_labels = batch
                input_ids, token_type_ids, lm_labels = \
                    input_ids.to(self.config['device']), token_type_ids.to(self.config['device']), lm_labels.to(self.config['device'])
                
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = lm_labels
                )
                
                loss, logits = outputs[0], outputs[1]
                
                valid_losses.append(loss.item())
                valid_ppls.append(torch.exp(loss).item())
              
            valid_loss = np.mean(valid_losses)
            valid_ppl = np.mean(valid_ppls)
              
        return valid_loss, valid_ppl
        
              
    def inference(self):
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{self.config['end_command']}\".")
        self.model.eval()
        
        with torch.no_grad():
            cur_speaker = 1
            input_ids_list = []
            token_type_ids_list = []
            t = 0
            output_id = None
            
            while True:
                if cur_speaker == 1:
                    cur_speaker_id = self.config['speaker1_id']
                    utter = input("You: ")
                    
                    if utter == self.config['end_command']:
                        print("Bot: Good bye.")
                        break
                    
                    input_id = [cur_speaker_id] + self.tokenizer.encode(utter)
                    
                    if t == 0:
                        input_id = [self.config['bos_id']] + input_id
                else:
                    cur_speaker_id = self.config['speaker2_id']
                    input_id = copy.deepcopy(output_id)
                    
                token_type_id = [cur_speaker_id] * len(input_id)
                
                if input_id[-1] == self.config['eos_id']:
                    input_id = input_id[:-1]
                    token_type_id = token_type_id[:-1] 
                
                input_ids_list.append(input_id)
                token_type_ids_list.append(token_type_id)
                
                if t >= self.config['max_time']:
                    input_ids_list = input_ids_list[1:]
                    token_type_ids_list = token_type_ids_list[1:]
                
                next_speaker = (cur_speaker % 2) + 1
                if next_speaker == 1:
                    next_speaker_id = self.config['speaker1_id']
                else:
                    next_speaker_id = self.config['speaker2_id']
                
                if cur_speaker == 1:
                    output_id = self.nucleus_sampling(input_ids_list, token_type_ids_list, next_speaker_id)
                    res = self.tokenizer.decode(output_id)

                    print(f"Bot: {res}")
                
                cur_speaker = copy.deepcopy(next_speaker)
                t += 1
                
    def nucleus_sampling(self, input_ids_list, token_type_ids_list, next_speaker_id):
        output_id = []
        res_id = [next_speaker_id]
        res_type_id = [next_speaker_id]
        for pos in range(self.config['utter_len']):
            input_ids = list(chain.from_iterable(input_ids_list)) + res_id
            token_type_ids = list(chain.from_iterable(token_type_ids_list)) + res_type_id
            input_len = len(input_ids)
            
            left = self.config['max_len'] - len(input_ids)
            input_ids += [self.config['pad_id']] * left
            token_type_ids += [self.config['pad_id']] * left

            assert len(input_ids) == len(token_type_ids), "There is something wrong in dialogue process."
            
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.config['device'])  # (1, L)
            token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.config['device'])  # (1, L)
            
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, input_len-1]  # (1, vocab_size)
            output = F.softmax(output, dim=-1)  # (1, vocab_size)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, vocab_size)
            idx_remove = cumsum_probs > self.config['nucleus_p']
            sorted_probs[idx_remove] = 1e-8
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, vocab_size)
            
            # Random sampling
            probs = torch.zeros(output.shape).to(self.config['device']).scatter_(-1, sorted_idxs, sorted_probs)  # (1, vocab_size)
            idx = torch.multinomial(probs, 1).squeeze(-1).squeeze(0).item()
            
            if len(output_id) == self.config['utter_len'] or idx == self.config['eos_id']:
                break
            else:
                output_id.append(idx)
                res_id.append(idx)
                res_type_id.append(next_speaker_id)
                
        return output_id
                    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help="The path to configuration file.")
    parser.add_argument('--mode', required=True, help="Train or inference?")
    parser.add_argument('--ckpt_name', required=False, help="Best checkpoint file.")
              
    args = parser.parse_args()
    
    assert args.mode == 'train' or args.mode=='inference', print("Please specify a correct mode name, 'train' or 'inference'.")
              
    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(args.config_path, args.mode, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(args.config_path, args.mode)
              
        manager.train()
        
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
        
        manager = Manager(args.config_path, args.mode, ckpt_name=args.ckpt_name)
        
        manager.inference()
