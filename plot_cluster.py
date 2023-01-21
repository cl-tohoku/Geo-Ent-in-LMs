import argparse
import io,sys
import os
import csv
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import re
from scipy.spatial.distance import cosine
from plot import plot_embeddings_bokeh
from plot import plot_embeddings
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from tqdm import tqdm
import collections
from itertools import chain
import japanize_matplotlib
import seaborn as sns
import math


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True, nargs='*', 
                            help="input dataset path")
    parser.add_argument("--emb_path", required=True, nargs='*', 
                            help="")
    #parser.add_argument("--ave_emb_path", required=True, nargs='*', 
    #                        help="")
    parser.add_argument("--output_path", required=True, type=os.path.abspath, 
                            help="output path")
    parser.add_argument("--is_group_by_Wiki_id", action='store_true',
                            help="")
    args = parser.parse_args()
    return args



def print_arg_path_list(path_list):
    print(path_list)
    print(f'path_len: {len(path_list)}')
    print()

def multiple_read_jsonl(path_list):
    print("multiple_read_jsonl")
    df_list = []
    for path in path_list:
        print(f'loading : {path}')
        df_list.append(pd.read_json(path, orient="records", lines=True))
    print()
    return df_list

def multiple_load_tensor(path_list):
    print("multiple_load_tensor")
    #torch_list = []

    for i, path in enumerate(path_list):
        print(f'loading : {path}')
        if i == 0:
            emb = torch.load(path)
        elif i>=1:
            emb_tmp = torch.load(path)
            emb = torch.cat((emb, emb_tmp), dim=0)

    print()
    return emb

def generate_average_vector(embeddings):
    """
    embeddings: list or ndarray or torch.tensor,  size of ([n,768]) 
    output : torch.tensor
    """
    #if torch.is_tensor(embeddings) == False:
    #    embeddings = torch.tensor(embeddings)
    average_vector = torch.sum(embeddings, axis=0) / len(embeddings)
    return average_vector

def aggregate_df(df, is_group_by_Wiki_id=False):
    print(f"is_group_by_Wiki_id : {is_group_by_Wiki_id}")
    ## 処理用データ作成
    print('データを集約')
    sentence_dict = defaultdict(list)
    target_word_dict = defaultdict(list)
    target_word_embeddings_dict = defaultdict(list)
    word_type_dict = {}
    category_dict = {}
    wiki_id_dict = {}
    target_word_sub_len_dict = {}
    alias_count_dict = {}
    word_type_dict = {}
    
    print(df.columns)
    for i , (target_word, sentence, target_word_embedding, word_type) in enumerate(zip(tqdm(df['target_word']), df['sentence'], df['target_word_embeddings_list'], df['word_type'])):
        if is_group_by_Wiki_id: # group by wiki_id
            if word_type == 'ne':
                sentence_dict[df['wiki_id'][i]].append(sentence)
                target_word_dict[df['wiki_id'][i]].append(target_word)
                target_word_embeddings_dict[df['wiki_id'][i]].append(target_word_embedding)
                word_type_dict[df['wiki_id'][i]] = word_type
                if 'notable_figer_types'  in df.columns:
                    category_dict[df['wiki_id'][i]] = df['notable_figer_types'][i]
                if 'wiki_id'  in df.columns:
                    wiki_id_dict[df['wiki_id'][i]] = df['wiki_id'][i]
                if 'target_word_sub_len'  in df.columns:
                    target_word_sub_len_dict[df['wiki_id'][i]] = df['target_word_sub_len'][i]
                if 'alias_count' in df.columns:
                    alias_count_dict[df['wiki_id'][i]] = df['alias_count'][i]

            elif word_type == 'non_ne':
                sentence_dict[target_word].append(sentence)
                target_word_embeddings_dict[target_word].append(target_word_embedding)
                target_word_dict[target_word].append(target_word)
                word_type_dict[target_word] = word_type
                if 'target_word_sub_len'  in df.columns:
                    target_word_sub_len_dict[target_word] = df['target_word_sub_len'][i]
                if 'notable_figer_types'  in df.columns:
                    category_dict[target_word] = df['notable_figer_types'][i]
                if 'wiki_id'  in df.columns:
                    wiki_id_dict[target_word] = df['wiki_id'][i]
                if 'target_word_sub_len'  in df.columns:
                    target_word_sub_len_dict[target_word] = df['target_word_sub_len'][i]
                if 'alias_count'  in df.columns:
                    alias_count_dict[target_word] = df['alias_count'][i]

        else: # group by target_word
            sentence_dict[target_word].append(sentence)
            target_word_embeddings_dict[target_word].append(target_word_embedding)
            word_type_dict[target_word] = word_type
            if 'notable_figer_types'  in df.columns:
                category_dict[target_word] = df['notable_figer_types'][i]
            if 'wiki_id'  in df.columns:
                wiki_id_dict[target_word] = df['wiki_id'][i]
            if 'target_word_sub_len'  in df.columns:
                target_word_sub_len_dict[target_word] = df['target_word_sub_len'][i]


        #target_word_embeddings_dict[target_word].append(torch.stack(target_word_embedding, dim=0))

    print('sentence_count 作成')
    sentence_count = [len(s) for s in list(sentence_dict.values())]

    
    ## df作成
    aggregated_df = pd.DataFrame(
        data = {
                'sentence_list': list(sentence_dict.values()),
                'sentence_count': sentence_count,
                'target_word_embeddings_list' : list(target_word_embeddings_dict.values()),
                'word_type' : list(word_type_dict.values())
                }
        )
    if is_group_by_Wiki_id: # group by wiki_id
        aggregated_df['target_word'] = list(target_word_dict.values())
    else: # group by target_word
        aggregated_df['target_word'] = list(sentence_dict.keys())

    if 'notable_figer_types'  in df.columns:
        aggregated_df['notable_figer_types'] = list(category_dict.values())
    if 'wiki_id'  in df.columns:
        aggregated_df['wiki_id'] = list(wiki_id_dict.values())
    if 'target_word_sub_len'  in df.columns:
        aggregated_df['target_word_sub_len'] = list(target_word_sub_len_dict.values())
    if 'alias_count'  in df.columns:
        aggregated_df['alias_count'] = list(alias_count_dict.values())
    
    print(aggregated_df.head())
    print(f"センテンス数： {aggregated_df['sentence_count'].sum()}")
    print(f"target_word数： {len(aggregated_df['target_word'])}\n")

    return aggregated_df




args = get_args()

## dataset install
df_list = multiple_read_jsonl(args.jsonl_path)
emb = multiple_load_tensor(args.emb_path)


## splitされたデータをconcatする
concat_df = pd.concat(df_list).reset_index(drop=True)
print(concat_df['notable_figer_types'])
#concat_df['notable_figer_types'] = [types[0] for types in concat_df['notable_figer_types']]


print(f'len(concat_emb): {len(emb)}','\n')


#concat_df['target_word_embeddings_list'] = emb
#concat_df['target_word_embeddings_list'] = torch.stack(emb, dim=0)

print(concat_df.head())
print(f"dataframe keys : {concat_df.keys()}")

# dfを集約する
#aggregated_df = aggregate_df(concat_df, args.is_group_by_Wiki_id)


## カテゴリ list作成
super_category_list = []
sub_category_list = []
for notable_figer_types in concat_df['notable_figer_types']:
    if pd.isna(notable_figer_types):
        super_category = 'Non NE' 
        sub_category = 'Non NE' 
    elif '/person' in notable_figer_types[0]:
        super_category = 'Person'
        sub_category = notable_figer_types[0]
    elif '/location' in notable_figer_types[0]:
        super_category = 'Location'
        sub_category = notable_figer_types[0]
    super_category_list.append(super_category)
    sub_category_list.append(sub_category)

super_category_list = np.array(super_category_list)
sub_category_list = np.array(sub_category_list)

class2color = {'Non NE':'#ff7f0e', 'Person':'#1f77b4', 'Location':'#2ca02c'}

print(f"output path : {args.output_path}")
plot_embeddings(emb, outfile=args.output_path, emb_method="UMAP", max_labels=len(emb), classes=super_category_list, class2color=class2color)

