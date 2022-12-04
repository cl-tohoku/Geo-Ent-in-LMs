import argparse
import io,sys
import os
import csv
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine
from plot import plot_embeddings_bokeh
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from tqdm import tqdm
from bokeh.palettes import Category20
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=os.path.abspath, 
                            help="input dataset path")
    # 後にcategoryは廃止する
    parser.add_argument('--category', type=str, default='ne',
                        help='word category. e.g. ne, common_noun')
    parser.add_argument("--name_type", type=str, default='lastname',
                        help="using (frist name, last name) pair or (last name)")
    parser.add_argument("--L_p", type=int, default=2,
                        help="Setting the L_p norm used in the distance function")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="")
    #parser.add_argument("--sentences_lower", type=int, default=0,
    #                    help="Lower limit on the number of sentences")
    #parser.add_argument("--split", type=int, default=0,
    #                    help="data split")
    args = parser.parse_args()
    return args

# TODO: target_wordに後に変更する
def get_target_token_category(args):
    if args.category == 'ne':
        if args.name_type == 'lastname':
            return 'lastname'
        elif args.name_type == 'firstname_lastname':
            return 'target_ne'
        else :
            raise ValueError("Please enter first and last name pairs or last names for analysis\n e.g. \'--name_type lastname\' or  \'--name_type firstname_lastname\'")
    elif args.category == 'large_ne':
        return 'target_ne'
    elif args.category == 'common_noun':
        return 'noun'
    elif args.category == 'test':
        return 'target_ne'
    else :
        raise ValueError("Please enter the word category you want to analyze.\n e.g. \'--category ne\' or \'--category common_noun\'")


def generate_average_vector(embeddings):
    """
    embeddings: list or ndarray or torch.tensor,  size of ([n,768]) 
    output : torch.tensor
    """
    if torch.is_tensor(embeddings) == False:
        embeddings = torch.tensor(embeddings)
    average_vector = torch.sum(embeddings, axis=0) / len(embeddings)
    return average_vector


def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        np.array: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        #hidden_states = outputs[2][1:]
        token_embeddings = outputs.last_hidden_state

    # Getting embeddings from the final BERT layer
    #token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = np.array([token_embed.tolist() for token_embed in token_embeddings])

    return list_token_embeddings


def plot_embeddings(df):
    bokeh_target_token_embeddings = []
    colors = []
    classes = []
    label_texts = []
    colors_category_list = Category20[20]
    cnt = 0
    for target_token, sentence_list, emb, sentence_count in zip(df[target_token_category], df['sentence_list'], df['target_token_embeddings_list'], df['sentence_count']):
        bokeh_target_token_embeddings.extend(np.array(emb))
        classes += ([target_token] * sentence_count)
        colors += ([colors_category_list[cnt]] * sentence_count)
        for sentence in sentence_list:
            label_texts.append(sentence)
        cnt += 1
        # ラベル数が多すぎると，色がわからなくなるので，プロットするカテゴリ数の最大は20個としている
        if cnt >= 20 :
            break
    plot_embeddings_bokeh(bokeh_target_token_embeddings, emb_method="UMAP", labels=label_texts, classes=classes, color=colors, size=8)

def cal_micro_ave(list_1, list_2):
    return list_1.sum() / list_2.sum()

def save_df_to_csv(df, input_path):
    # result dir がなければ作成する
    basename_without_ext = os.path.splitext(os.path.basename(str(input_path)))[0]
    savefile = "./result/" + basename_without_ext +"_percentage_of_own_cluster.csv"
    print(f'savefile path {savefile}')

    output_df = pd.DataFrame([df[target_token_category], df['percentage_of_own_cluster'], df['own_count_list'],  df['sentence_count']]).T
    micro_ave = cal_micro_ave(df['own_count_list'], df['sentence_count'])
    print(f'micro_ave : {micro_ave}')
    output_df.append({"micro_ave" : micro_ave}, ignore_index=True)
    output_df.to_csv(savefile, encoding="utf_8_sig")


class MyDataset(Dataset):
    def __init__(self, path):
        #self.csv_df = pd.read_csv(path)
        self.df = pd.read_json(jsonl_file_path, orient="records", lines=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target_word = self.df['target_word'][idx]
        sentence =  self.df['sentence'][idx]
        return target_word, sentence


# Loading the pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
# Setting up the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

args = get_args()

## dataset install
jsonl_file_path = args.input
#print(f'jsonl_file_path : {jsonl_file_path}')
is_file = os.path.isfile(jsonl_file_path)
if is_file == False:
    print(f"{jsonl_file_path} が存在しません")
    raise ValueError()
sentence_dataset = MyDataset(jsonl_file_path)
dataloader = DataLoader(sentence_dataset, batch_size=args.batch_size)

print(f'input file path: {jsonl_file_path}')
print(f'batch_size: {args.batch_size}')


# modelへの最大入力長
INPUT_MAX_LENGTH = 512

# TODO: Embedding保存済みのdfをjsonlとして保存→以降それがあれば，それをdfとしてインストール&Embedding計算を省く
# 現状は未実装
cached_jsonl_file_path = '/data/cached_file_path.jsonl'
if os.path.isfile(cached_jsonl_file_path):
    pass
    #df = pd.read_json(cached_jsonl_file_path, orient='records', lines=True)
else:
    #df = pd.read_json(jsonl_file_path, orient="records", lines=True)
    #print(df.head())
    print("Computing Embedding")
    # Getting embeddings for the target word
    # word in all given contexts

    #target_token_category = get_target_token_category(args)

    target_token_embeddings_list = []
    for i, data in enumerate(tqdm(dataloader)):
        target_word, sentence = data
        tokenized_target_word = tokenizer(target_word, add_special_tokens=False)
        tokenized_sentence = tokenizer(sentence, return_tensors="pt", max_length=INPUT_MAX_LENGTH, padding=True, truncation=True)
        tokenized_sentence_list = tokenized_sentence['input_ids'].tolist()
        with torch.no_grad():
            outputs = model(**tokenized_sentence)
            last_hidden_states = outputs.last_hidden_state
        try:
            # target_wordのindexを得る 
            # いまのところ多分サブワードは考慮していない（サブワード分割の先頭のみ）
            # TODO:↓ここfor文に書き直す&tryでチェックして，ValueErrorがあれば，所定の場所のみ[] か""にする?
            tokenized_target_token_indexes = [tokenized_s.index(tokenized_target_w[0]) for (tokenized_target_w, tokenized_s) in [(tokenized_target_w, tokenized_s) for tokenized_target_w, tokenized_s in zip(tokenized_target_word['input_ids'], tokenized_sentence_list)]]
            tokenized_target_token_embeddings = [sentence_embeddings[token_index] for token_index, sentence_embeddings in zip(tokenized_target_token_indexes, last_hidden_states)]
        except Exception as error:
            print(error)
            for w, s in zip(target_word, sentence):
                print(tokenizer.tokenize(w))
                print(tokenizer.tokenize(s))
            raise error  
            #print(f'sentence_list index : {i}')
            #print(f'target_word : {target_word}')
            #print(f'sentence : {sentence}')
             
        
        target_token_embeddings_list.extend(tokenized_target_token_embeddings)
        emb_tensor = torch.stack(target_token_embeddings_list, dim=0)

    # save embeddings
    path_without_ext = str(jsonl_file_path).replace('.jsonl', '')
    print(f"save: {path_without_ext}_tensor.pt")
    torch.save(emb_tensor, path_without_ext + '_tensor.pt')


    # df →　jsonl形式で保存する
    #df.to_json('/data/wikilinks/cached_testdata.jsonl', orient='records', force_ascii=False, lines=True)
    #float はjsonで保存できないっぽい

