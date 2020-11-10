from transformers import *
from custom_data import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F

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
            'unk_token': self.config['unk'],
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
        
        # Load model    
        print("Loading the model...")
        self.model = GPT2DoubleHeadsModel.from_pretrained('gpt2').to(self.config['device'])
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
            lm_train_losses = []
            mc_train_losses = []
            total_train_losses = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, attention_masks, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
                input_ids, attention_masks, token_type_ids, mc_token_ids, lm_labels, mc_labels = \
                    input_ids.to(self.config['device']), attention_masks.to(self.config['device']), token_type_ids.to(self.config['device']), \
                    mc_token_ids.to(self.config['device']), lm_labels.to(self.config['device']), mc_labels.to(self.config['device'])
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask = attention_masks,
                    token_type_ids = token_type_ids,
                    mc_token_ids = mc_token_ids,
                    labels = lm_labels,
                    mc_labels = mc_labels
                )
                
                lm_loss, mc_loss = outputs[0], outputs[1]
                loss = self.config['lm_coef'] * lm_loss + self.config['mc_coef']
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                lm_train_losses.append(lm_loss.item())
                mc_train_losses.append(mc_loss.item())
                total_train_losses.append(loss.item())
            
            lm_train_loss = np.mean(lm_train_losses)
            mc_train_loss = np.mean(mc_train_losses)
            total_train_loss = np.mean(total_train_losses)
            print(f"Train loss: {total_train_loss} || LM loss: {lm_train_loss} || MC loss: {mc_train_loss}")
            
            lm_valid_loss, mc_valid_loss, valid_loss = self.validation()
              
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
            print(f"Valid loss: {valid_loss}")
              
        print("Training finished!")
    
    def validation(self):
        print("Validation processing...")
        self.model.eval()
              
        lm_valid_losses = []
        mc_valid_losses = []
        total_valid_losses = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, attention_masks, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
                input_ids, attention_masks, token_type_ids, mc_token_ids, lm_labels, mc_labels = \
                    input_ids.to(self.config['device']), attention_masks.to(self.config['device']), token_type_ids.to(self.config['device']), \
                    mc_token_ids.to(self.config['device']), lm_labels.to(self.config['device']), mc_labels.to(self.config['device'])
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask = attention_masks,
                    token_type_ids = token_type_ids,
                    mc_token_ids = mc_token_ids,
                    labels = lm_labels,
                    mc_labels = mc_labels
                )
                
                lm_loss, mc_loss = outputs[0], outputs[1]
                loss = self.config['lm_coef'] * lm_loss + self.config['mc_coef']
                
                lm_valid_losses.append(lm_loss.item())
                mc_valid_losses.append(mc_loss.item())
                total_valid_losses.append(loss.item())
              
            lm_valid_loss = np.mean(lm_valid_losses)
            mc_valid_loss = np.mean(mc_valid_losses)
            total_valid_loss = np.mean(total_valid_losses)
              
        return lm_valid_loss, mc_valid_loss, total_valid_loss
        
              
    def inference(self):
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{self.config['end_command']}\".")
        self.model.eval()
        

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
