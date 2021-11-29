from tqdm import tqdm
from datasets import *


# For all
space = 'Ġ'
pre_quote = '’'
end_marks = ['.', ',', '?', '!', '...']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']

# For empathetic dialogues
exclude_symbol = "_conv"
comma_symbol = "_comma_"

# For persona chat
persona_chat_url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
silence_symbol = "__ SILENCE __"


def load_daily(tokenizer, train_frac):
    dataset = load_dataset('daily_dialog')
    train_dialogues = dataset['train']['dialog']
    valid_dialogues = dataset['validation']['dialog']
    test_dialogues = dataset['test']['dialog']
    
    total_dialogues = train_dialogues + valid_dialogues + test_dialogues
    
    for i, dialogue in enumerate(tqdm(total_dialogues)):
        new_dialogue = []
        for utter in dialogue:
            token_list = tokenizer.tokenize(utter.strip().replace(pre_quote, quotes[1]))
            token_list = process_token_list(token_list)
            text = tokenizer.convert_tokens_to_string(token_list)
            new_dialogue.append(text)
            
        total_dialogues[i] = new_dialogue
    
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num
    
    
def load_empathetic(tokenizer, train_frac):
    dataset = load_dataset('empathetic_dialogues')
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']
    
    total_utters = train_data['utterance'] + valid_data['utterance'] + test_data['utterance']
    total_conv_ids = train_data['conv_id'] + valid_data['conv_id'] + test_data['conv_id']
    total_speaker_ids = train_data['speaker_idx'] + valid_data['speaker_idx'] + test_data['speaker_idx']
    
    assert len(total_utters) == len(total_conv_ids) and len(total_conv_ids) == len(total_speaker_ids)
    
    num = 0
    
    conv_dict = {}
    cur_speaker_idx = -1
    for i, utter in enumerate(tqdm(total_utters)):
        conv_id = total_conv_ids[i]
        speaker_idx = total_speaker_ids[i]
        
        utter_modified = utter.strip().replace(comma_symbol, ',')
        new_token_list = process_token_list(tokenizer.tokenize(utter_modified))
        text = tokenizer.convert_tokens_to_string(new_token_list)
        
        if exclude_symbol in utter:
            continue
        
        if conv_id not in conv_dict:
            conv_dict[conv_id] = []
            cur_speaker_idx = -1

        if cur_speaker_idx != speaker_idx:
            conv_dict[conv_id].append(text)
            cur_speaker_idx = speaker_idx
        else:
            conv_dict[conv_id][-1] += f" {text}"
    
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = []
    valid_dialogues = []
    
    train_dialogue_num = int(len(conv_dict) * train_frac)
    for i, (conv_id, utter_list) in enumerate(conv_dict.items()):
        if i < train_dialogue_num:
            train_utter_num += len(utter_list)
            train_dialogues.append(utter_list)
        else:
            valid_utter_num += len(utter_list)
            valid_dialogues.append(utter_list)
            
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num


def load_persona(tokenizer, train_frac):
    import urllib.request, json
    with urllib.request.urlopen(persona_chat_url) as f:
        dataset = json.loads(f.read().decode())
        
    train_data = dataset['train']
    valid_data = dataset['valid']
    total_data = train_data + valid_data
    total_dialogues = []
    
    for obj in tqdm(total_data):
        dialogue = obj['utterances'][-1]['history']
        new_dialogue = []
        
        for i, utter in enumerate(dialogue):
            if utter.strip() != silence_symbol:
                token_list = tokenizer.tokenize(utter.strip())
                new_token_list = process_token_list(token_list)
                text = tokenizer.convert_tokens_to_string(new_token_list)
                new_dialogue.append(text)
        
        total_dialogues.append(new_dialogue)
        
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num


def load_blended(tokenizer, train_frac):
    dataset = load_dataset('blended_skill_talk')
    data_train = dataset['train']
    data_valid = dataset['validation']
    data_test = dataset['test']
    
    total_previous_utterance = data_train['previous_utterance'] + data_valid['previous_utterance'] + data_test['previous_utterance']
    total_free_messages = data_train['free_messages'] + data_valid['free_messages'] + data_test['free_messages']
    total_guided_messages = data_train['guided_messages'] + data_valid['guided_messages'] + data_test['guided_messages']

    total_dialogues = []
    for i, free_message in enumerate(tqdm(total_free_messages)):
        free_message_list = [utter.strip() for utter in free_message if len(utter.strip())>0]
        guided_message_list = [utter.strip() for utter in total_guided_messages[i] if len(utter.strip())>0]
        dialogue = total_previous_utterance[i]
        
        for j in range(len(free_message_list)):
            token_list = process_token_list(tokenizer.tokenize(free_message_list[j]))
            text = tokenizer.convert_tokens_to_string(token_list)
            dialogue.append(text)
            
            if j < len(guided_message_list):
                token_list = process_token_list(tokenizer.tokenize(guided_message_list[j]))
                text = tokenizer.convert_tokens_to_string(token_list)
                dialogue.append(text)
            
        total_dialogues.append(dialogue)
        
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num
    

def process_token_list(token_list):
    token_list[0] = token_list[0].capitalize()
    
    quote_count = 0
    for i, token in enumerate(token_list):
        if space in token:
            if token[1:] in end_marks or token[1:] in abbreviations:
                token_list[i] = token[1:]
                
            if token[1:] == quotes[1]:
                if i<len(token_list)-1:
                    if token_list[i+1] in abbreviations or (token_list[i+1][0] == space and token_list[i+1][1:] in abbreviations):
                        token_list[i] = token[1:]
                        
        if token[0] == space and token[1:] in quotes:
            if quote_count % 2 == 1:
                token_list[i] = token[1:]
                quote_count = 0
            else:
                if i<len(token_list)-1 and token_list[i+1][0] == space:
                    token_list[i+1] = token_list[i+1][1:]
                quote_count += 1
                
        if token in end_marks or token[1:] in end_marks:
            if i<len(token_list)-1:
                if token_list[i+1][0] != space:
                    token_list[i+1] = space + token_list[i+1].capitalize()
                else:
                    token_list[i+1] = space + token_list[i+1][1:].capitalize()
                
    new_token_list = [token for token in token_list if token != space and len(token)>0]
    if new_token_list[-1] not in end_marks:
        new_token_list.append(end_marks[0])
        
    return new_token_list
