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
import copy
import random


class Manager():
    def __init__(self, args):
        self.args = args
        
        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            'bos_token': self.args.bos_token,
            'eos_token': self.args.eos_token,
            'pad_token': self.args.pad_token,
            'additional_special_tokens': [self.args.sp1_token, self.args.sp2_token]
        }
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.pad_id = vocab[self.args.pad_token]
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        
        self.args.utter_len = (self.args.max_len-self.args.max_turns-2) // self.args.max_turns
        
        # Load model    
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
            
        if self.args.mode == 'train':            
            # Load optimizer
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            self.best_loss = sys.float_info.max
            self.last_epoch = 0
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset(self.args.train_prefix, self.args)
            valid_set = CustomDataset(self.args.valid_prefix, self.args)
            ppd = PadCollate(pad_id=self.args.pad_id)
            
            self.fix_seed(self.args.seed)
            self.train_loader = DataLoader(train_set, 
                                           collate_fn=ppd.pad_collate, 
                                           shuffle=True, 
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            self.valid_loader = DataLoader(valid_set, 
                                           collate_fn=ppd.pad_collate,
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            
            if not os.path.exists(self.args.ckpt_dir):
                os.makedirs(self.args.ckpt_dir)
        
        if self.args.ckpt_name is not None:
            if os.path.exists(f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt"):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt")
                self.model.load_state_dict(ckpt['model_state_dict'])
                
                if self.args.mode == 'train':
                    print(f"The training restarts with the specified checkpoint: {self.args.ckpt_name}.ckpt.")
                    self.optim.load_state_dict(ckpt['optim_state_dict'])
                    self.best_loss = ckpt['loss']
                    self.last_epoch = ckpt['epoch']
                else:
                    print("The inference will start with the specified checkpoint.")
            else:
                print("Cannot fine the specified checkpoint.")
                if self.args.mode == 'train':
                    print("Training will start with the initialized model.")
                else:
                    print("Cannot inference.")
                    exit()
              
        print("Setting finished.")
              
    def train(self):
        print("Training starts.")
        
        start_epoch = self.last_epoch+1
        for epoch in range(start_epoch, start_epoch+self.args.num_epochs):
            self.model.train()
            
            print(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
            train_losses = []
            train_ppls = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, lm_labels = batch
                input_ids, token_type_ids, lm_labels = \
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), lm_labels.to(self.args.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = lm_labels
                )
                
                loss, logits = outputs[0], outputs[1]
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                train_losses.append(loss)
                train_ppls.append(torch.exp(loss))
            
            train_losses = [loss.item() for loss in train_losses]
            train_ppls = [ppl.item() for ppl in train_ppls]
            train_loss = np.mean(train_losses)
            train_ppl = np.mean(train_ppls)
            print(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")
            
            self.last_epoch += 1
            
            valid_loss, valid_ppl = self.validation()
              
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': self.best_loss,
                    'epoch': self.last_epoch
                }
              
                torch.save(state_dict, f"{self.args.ckpt_dir}/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
                print("*"*10 + "Current best checkpoint is saved." + "*"*10)
                print(f"{self.args.ckpt_dir}/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
              
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
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), lm_labels.to(self.args.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = lm_labels
                )
                
                loss, logits = outputs[0], outputs[1]
                
                valid_losses.append(loss)
                valid_ppls.append(torch.exp(loss))
            
            valid_losses = [loss.item() for loss in valid_losses]
            valid_ppls = [ppl.item() for ppl in valid_ppls]
            valid_loss = np.mean(valid_losses)
            valid_ppl = np.mean(valid_ppls)
              
        return valid_loss, valid_ppl
        
              
    def inference(self):
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{self.args.end_command}\".")
        self.model.eval()
        
        with torch.no_grad():
            cur_sp = 1
            input_ids_list = []
            token_type_ids_list = []
            t = 0
            output_id = None
            
            while True:
                if cur_sp == 1:
                    cur_sp_id = self.args.sp1_id
                    utter = input("You: ")
                    
                    if utter == self.args.end_command:
                        print("Bot: Good bye.")
                        break
                    
                    input_id = [cur_sp_id] + self.tokenizer.encode(utter)
                    
                    if t == 0:
                        input_id = [self.args.bos_id] + input_id
                else:
                    cur_sp_id = self.args.sp2_id
                    input_id = copy.deepcopy(output_id)
                    
                token_type_id = [cur_sp_id] * len(input_id)
                
                if input_id[-1] == self.args.eos_id:
                    input_id = input_id[:-1]
                    token_type_id = token_type_id[:-1] 
                
                input_ids_list.append(input_id)
                token_type_ids_list.append(token_type_id)
                
                if t >= self.args.max_turns:
                    input_ids_list = input_ids_list[1:]
                    token_type_ids_list = token_type_ids_list[1:]
                
                next_sp = (cur_sp % 2) + 1
                if next_sp == 1:
                    next_sp_id = self.args.sp1_id
                else:
                    next_sp_id = self.args.sp2_id
                
                if cur_sp == 1:
                    output_id = self.nucleus_sampling(input_ids_list, token_type_ids_list, next_sp_id)
                    res = self.tokenizer.decode(output_id)

                    print(f"Bot: {res}")
                
                cur_sp = next_sp
                t += 1
                
    def nucleus_sampling(self, input_ids_list, token_type_ids_list, next_sp_id):
        output_id = []
        res_id = [next_sp_id]
        res_type_id = [next_sp_id]
        for pos in range(self.args.utter_len):
            input_ids = list(chain.from_iterable(input_ids_list)) + res_id
            token_type_ids = list(chain.from_iterable(token_type_ids_list)) + res_type_id
            input_len = len(input_ids)
            
            left = self.args.max_len - len(input_ids)
            input_ids += [self.args.pad_id] * left
            token_type_ids += [self.args.pad_id] * left

            assert len(input_ids) == len(token_type_ids), "There is something wrong in dialogue process."
            
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args.device)  # (1, L)
            token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args.device)  # (1, L)
            
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, input_len-1]  # (1, vocab_size)
            output = F.softmax(output, dim=-1)  # (1, vocab_size)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, vocab_size)
            idx_remove = cumsum_probs > self.args.top_p
            sorted_probs[idx_remove] = 1e-8
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, vocab_size)
            
            # Random sampling
            probs = torch.zeros(output.shape).to(self.args.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, vocab_size)
            idx = torch.multinomial(probs, 1).squeeze(-1).squeeze(0).item()
            
            if len(output_id) == self.args.utter_len or idx == self.args.eos_id:
                break
            else:
                output_id.append(idx)
                res_id.append(idx)
                res_type_id.append(next_sp_id)
                
        return output_id
    
    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
                    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--mode', type=str, required=True, help="The running mode: train or inference?")
    parser.add_argument('--data_dir', type=str, default="data", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument('--model_type', type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument('--pad_token', type=str, default="<pad>", help="The pad token.")
    parser.add_argument('--bos_token', type=str, default="<bos>", help="The BOS token.")
    parser.add_argument('--eos_token', type=str, default="<eos>", help="The EOS token.")
    parser.add_argument('--sp1_token', type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument('--sp2_token', type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    parser.add_argument('--lr', type=float, default=5e-4, help="The learning rate.")
    parser.add_argument('--batch_size', type=int, default=8, help="The batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--num_epochs', type=int, default=10, help="The number of total epochs.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument('--max_turns', type=int, default=5, help="The maximum number of dialogue histories to include.")
    parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--ckpt_dir', type=str, default="saved_models", help="The directory name for saved checkpoints.")
    parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--end_command', type=str, default="Abort!", help="The command to stop the conversation when inferencing.")
              
    args = parser.parse_args()
    
    assert args.mode in ["train", "inference"]
    assert args.model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    
    args.data_dir = f"{args.data_dir}/{args.model_type}"
    args.ckpt_dir = f"{args.ckpt_dir}/{args.model_type}"
              
    if args.mode == 'train':
        manager = Manager(args)
        manager.train()
        
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
        
        manager = Manager(args)
        manager.inference()
