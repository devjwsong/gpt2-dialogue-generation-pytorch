from transformers import GPT2Tokenizer, GPT2LMHeadModel
from custom_dataset import *
from torch.nn import functional as F
from itertools import chain

import torch
import numpy as np
import argparse
import random


class Inferencer():
    def __init__(self, args):
        self.args = args
        
        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_path)
        special_tokens = self.tokenizer.special_tokens_map
        self.args.bos_token = special_tokens['bos_token']
        self.args.eos_token = special_tokens['eos_token']
        self.args.sp1_token = special_tokens['additional_special_tokens'][0]
        self.args.sp2_token = special_tokens['additional_special_tokens'][1]

        vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        
        # Load model    
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_path).to(self.args.device)
        
        self.args.max_len = self.model.config.n_ctx
              
        print("Setting finished.")
              
    def infer(self):
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{self.args.end_command}\".")
        self.model.eval()
        self.fix_seed(self.args.seed)
        
        with torch.no_grad():
            input_hists = []
            
            while True:
                utter = input("You: ")
                if utter == self.args.end_command:
                    print("Bot: Good bye.")
                    break
                
                input_ids = [self.args.sp1_id] + self.tokenizer.encode(utter)
                input_hists.append(input_ids)
                
                if len(input_hists) >= self.args.max_turns:
                    num_exceeded = len(input_hists) - self.args.max_turns + 1
                    input_hists = input_hists[num_exceeded:]
                    
                input_ids = [self.args.bos_id] + list(chain.from_iterable(input_hists)) + [self.args.sp2_id]
                start_sp_id = input_hists[0][0]
                next_sp_id = self.args.sp1_id if start_sp_id == self.args.sp2_id else self.args.sp2_id
                assert start_sp_id != next_sp_id
                token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
                assert len(token_type_ids) == len(input_hists)
                token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.args.sp2_id]
                assert len(input_ids) == len(token_type_ids)
                input_len = len(input_ids)
                
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args.device)
                token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args.device)
                
                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)                
                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                print(f"Bot: {res}")
                input_hists.append([self.args.sp2_id] + self.tokenizer.encode(res))
                
    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos-1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.args.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)
            
            probs = torch.zeros(output.shape, device=self.args.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)
            
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)
            
            if idx_item == self.args.eos_id:
                break
                
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.args.sp2_id]]).to(self.args.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape
            
        return output_ids
    
    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
                    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--model_path', type=str, required=True, help="The path to the model in HuggingFace Hub.")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    parser.add_argument('--max_turns', type=int, default=5, help="The maximum number of dialogue histories to include.")
    parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--end_command', type=str, default="Abort!", help="The command to stop the conversation when inferencing.")
              
    args = parser.parse_args()
        
    inferencer = Inferencer(args)
    inferencer.infer()
