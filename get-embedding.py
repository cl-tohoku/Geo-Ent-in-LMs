import argparse
import io,sys
import os
import csv
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine
import itertools
from sklearn import mixture
from tqdm import tqdm
from bokeh.palettes import Category20
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=os.path.abspath, 
                            help="input dataset path")
    parser.add_argument("--output", required=True ,  type=os.path.abspath, 
                            help="output tensor path")
    parser.add_argument("--name_type", type=str, default='lastname',
                        help="using (frist name, last name) pair or (last name)")
    parser.add_argument("--L_p", type=int, default=2,
                        help="Setting the L_p norm used in the distance function")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="")
    parser.add_argument("--layer", type=int, default=None,
                        help="getting layer") 
    parser.add_argument("--model", type=str, default=None,
                        help="using model")       
    parser.add_argument("--cuda_number", type=str, default=None,
                        help="cuda number")  
    parser.add_argument("--is_dont_getEmbedding",  action='store_true',
                        help="")  
    parser.add_argument("--is_ne",  action='store_true',
                        help="")              
    args = parser.parse_args()
    return args


def generate_average_vector(embeddings):
    """
    embeddings: list or ndarray or torch.tensor,  size of ([n,768]) 
    output : torch.tensor
    """
    if torch.is_tensor(embeddings) == False:
        embeddings = torch.tensor(embeddings)
    average_vector = torch.sum(embeddings, axis=0) / len(embeddings)
    return average_vector



class MyDataset(Dataset):
    def __init__(self, path, is_ne_span=False):
        self.df = pd.read_json(jsonl_file_path, orient="records", lines=True, encoding='utf-8')
        self.is_ne_span = is_ne_span

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target_word = self.df['target_word'][idx]
        sentence = self.df['sentence'][idx]
        if self.is_ne_span:
            ne_span = tuple(self.df['ne_span'][idx])
            return target_word, sentence, ne_span
        else:
            return target_word, sentence



args = get_args()

## device check
if args.cuda_number is not None:
    device = torch.device(f"cuda:{args.cuda_number}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")


if args.model is None:
    print("model : bert-base-uncased")
    from transformers import BertTokenizer, BertModel
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
else :
    print(f"model : {args.model}")
    if args.model.split('-')[0] == "bert" :
        from transformers import BertTokenizer, BertModel
        model = BertModel.from_pretrained(args.model, output_hidden_states = True)
        tokenizer = BertTokenizer.from_pretrained(args.model)
    elif args.model.split('-')[0] == "roberta"  :
        from transformers import RobertaTokenizer, RobertaModel
        model = RobertaModel.from_pretrained(args.model, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
    elif args.model.split('-')[0] == "albert"  :
        from transformers import AlbertTokenizer, AlbertModel
        model = AlbertModel.from_pretrained(args.model, output_hidden_states = True)
        tokenizer = AlbertTokenizer.from_pretrained(args.model) 
    elif args.model.split('-')[0] == "distilbert"  :
        from transformers import DistilBertTokenizer, DistilBertModel
        model = DistilBertModel.from_pretrained(args.model, output_hidden_states = True)
        tokenizer = DistilBertTokenizer.from_pretrained(args.model) 
    elif "luke-" in args.model  :
        from transformers import LukeTokenizer, LukeModel
        model = LukeModel.from_pretrained(args.model, output_hidden_states = True)
        tokenizer = LukeTokenizer.from_pretrained(args.model) 
    else:
        raise ValueError("args.model is an invalid value.")

model.to(device)


## dataset install
jsonl_file_path = args.input
is_file = os.path.isfile(jsonl_file_path)
if is_file == False:
    print(f"{jsonl_file_path} が存在しません")
    raise ValueError()

if "luke-" in args.model and args.is_ne:
    sentence_dataset = MyDataset(jsonl_file_path, is_ne_span=True)
else:
    sentence_dataset = MyDataset(jsonl_file_path)

dataloader = DataLoader(sentence_dataset, batch_size=args.batch_size)

print(f'input file path: {jsonl_file_path}')
print(f'batch_size: {args.batch_size}')
if args.layer is not None:
    print(f'Location of the layer to be acquired : {args.layer}')
else : 
    print(f'Location of the layer to be acquired : last_layer')




# modelへの最大入力長
INPUT_MAX_LENGTH = 512

# TODO: Embedding保存済みのdfをjsonlとして保存→以降それがあれば，それをdfとしてインストール&Embedding計算を省く
# 現状は未実装
cached_jsonl_file_path = '/data/cached_file_path.jsonl'
if os.path.isfile(cached_jsonl_file_path):
    pass
    #df = pd.read_json(cached_jsonl_file_path, orient='records', lines=True)
else:
    print("Computing Embedding")
    # Getting embeddings for the target word
    # word in all given contexts

    target_token_embeddings_list = []
    for i, data in enumerate(tqdm(dataloader)):
        if "luke-" in args.model and args.is_ne:
            target_word, sentence, entity_spans = data
        else:
            target_word, sentence = data

        if args.model is not None and (args.model.split('-')[0] == "roberta" or "luke-" in args.model):
            if "luke-" in args.model and args.is_ne:
                #print(target_word)
                #print(sentence)
                #print(entity_spans)
                target_word = [[t_word] for t_word in target_word]
                entity_spans = [s.tolist() for s in entity_spans]
                entity_spans = [[spans] for spans in zip(*entity_spans)]
                tokenized_sentence = tokenizer(sentence, entities=target_word, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt", max_length=INPUT_MAX_LENGTH, padding=True, truncation=True)
            else:
                tokenized_target_word = tokenizer(target_word, add_special_tokens=False, add_prefix_space=True)
                tokenized_sentence = tokenizer(sentence, return_tensors="pt", max_length=INPUT_MAX_LENGTH, padding=True, truncation=True, add_prefix_space=True)
        else:
            tokenized_target_word = tokenizer(target_word, add_special_tokens=False)
            tokenized_sentence = tokenizer(sentence, return_tensors="pt", max_length=INPUT_MAX_LENGTH, padding=True, truncation=True)
        

        tokenized_sentence_list = tokenized_sentence['input_ids'].tolist()
        tokenized_sentence = tokenized_sentence.to(device)
        if args.is_dont_getEmbedding == False:
            with torch.no_grad():
                outputs = model(**tokenized_sentence)
                if args.layer is not None:
                    if "luke-" in args.model and args.is_ne:
                        hidden_states = outputs.entity_hidden_states[args.layer]
                    else:
                        hidden_states = outputs.hidden_states[args.layer] #output.hidden_states[0] is the input state (Therefore, ignore it)
                else :
                    if "luke-" in args.model and args.is_ne:
                        hidden_states = outputs.entity_last_hidden_state
                    else:
                        hidden_states = outputs.last_hidden_state
        try:
            # Get the embedding for target_token
            if "luke-" in args.model and args.is_ne:
                pass
            else:
                tokenized_target_token_indexes = [list(range(tokenized_s.index(tokenized_target_w[0]), tokenized_s.index(tokenized_target_w[0])+len(tokenized_target_w))) for (tokenized_target_w, tokenized_s) in [(tokenized_target_w, tokenized_s) for tokenized_target_w, tokenized_s in zip(tokenized_target_word['input_ids'], tokenized_sentence_list)]]
                if args.is_dont_getEmbedding == False:
                    tokenized_target_token_embeddings = [sentence_embeddings[token_index] for token_index, sentence_embeddings in zip(tokenized_target_token_indexes, hidden_states)]
                    ave_tokenized_target_token_embeddings = [generate_average_vector(embs) for embs in tokenized_target_token_embeddings]
        except Exception as error:
            print(error)
            print(f"sentence : {sentence}")
            print(f"target_word : {target_word}")
            raise error  
        if args.is_dont_getEmbedding == False:
            if "luke-" in args.model and args.is_ne:
                entity_hidden_states = [h[0] for h in hidden_states]
                target_token_embeddings_list.extend(entity_hidden_states)
            else:
                target_token_embeddings_list.extend(ave_tokenized_target_token_embeddings)
    
    if args.is_dont_getEmbedding == False:
        emb_tensor = torch.stack(target_token_embeddings_list, dim=0)
        # save embeddings
        dirname = os.path.dirname(args.output)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        print(f"save: {args.output}")
        torch.save(emb_tensor.to('cpu'), args.output)

