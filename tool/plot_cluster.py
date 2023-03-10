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
    parser.add_argument("--is_plot_nearest_cluster", action='store_true',
                            help="")
    parser.add_argument("--plot_target_word", type=str, default=None,
                            help="")
    parser.add_argument("--numberOfClusters", type=int, default=11,
                            help="")
    parser.add_argument("--marker_size", type=int, default=14,
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

def make_category2marker(nearest_cluster_df):
    category2marker = {}
    for target_word , notable_figer_types, word_type in zip(nearest_cluster_df['target_word'], nearest_cluster_df['notable_figer_types'], nearest_cluster_df['word_type']):
        if type(target_word) == list:
            target_w = target_word[0]
        elif type(target_word) == str:
            target_w = target_word

        if word_type == 'non_ne': # Non_NE -> 'v'
          category2marker[target_w] = 'v'
        elif '/person' in notable_figer_types: # Person -> 'o'
          category2marker[target_w] = 'o'
        elif '/location' in notable_figer_types: # Location -> 's'
          category2marker[target_w] = 's'

    return category2marker



def make_category2color(nearest_cluster_df, target_cluster_name):
    cmap = plt.get_cmap("Paired")
    category2marker = {}
    if type(nearest_cluster_df['target_word'].iloc[0]) == list:
        nearest_cluster_df['target_word'] = [target_word[0] for target_word in nearest_cluster_df['target_word']]


    if '/person' in nearest_cluster_df[nearest_cluster_df['target_word'] == target_cluster_name]['notable_figer_types'].values[0]: 
        target_category = 'person'
    elif '/location' in nearest_cluster_df[nearest_cluster_df['target_word'] == target_cluster_name]['notable_figer_types'].values[0]: 
        target_category = 'location'
    else: # Non_NE -> 'v'
        target_category = 'non_ne'

    target_word_uniquelist = list(set(nearest_cluster_df['target_word']))

    for target_word , notable_figer_types, word_type in zip(nearest_cluster_df['target_word'], nearest_cluster_df['notable_figer_types'], nearest_cluster_df['word_type']):
        
        if word_type == 'non_ne' and target_category == 'non_ne':
            category2marker[target_word] = cmap(target_word_uniquelist.index(target_word))
        elif word_type == 'ne' and target_category in notable_figer_types: 
            category2marker[target_word] = cmap(target_word_uniquelist.index(target_word))
        else: 
            category2marker[target_word] = 'black'
    return category2marker



def get_nearest_cluster_df(df, plot_target_word, p=2, numberOfClusters=11,  is_group_by_Wiki_id=False):
    """
    input:
        all clusters : dataframe
        target_word :str
    
    Returns:
        numberOfClusters  nearest neighbour clusters : dataframe
    """

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")

    # distance function
    pdist = torch.nn.PairwiseDistance(p=p)
    own_count = 0
    own_count_list = []
    dist_list = []
    percentage_of_own_cluster = []
    wrong_cluster_list = []
    wrong_type_list = []
    wrong_pair_list = []
    
    plot_target_index = 0
    if type(df['target_word'][0]) == list:
      for i, target_word_list in enumerate(df['target_word']):
        for target_word in target_word_list:
          if target_word == plot_target_word:
            plot_target_index = i
    elif type(df['target_word'][0]) == str:
        for i, target_word in enumerate(df['target_word']):
          if target_word == plot_target_word:
            plot_target_index = i


    #for i, (target_word_embeddings_list, sentence_count) in enumerate(zip(tqdm(df['target_word_embeddings_list']), df['sentence_count'])):
    #target_word_embeddings_list = df.iloc[plot_target_index]['target_word_embeddings_list']
    #sentence_count = df.iloc[plot_target_index]['sentence_count']

    #target_word_embeddings_tensor = torch.stack(target_word_embeddings_list).to(device)
    target_cluster_average_embeddings = df.iloc[plot_target_index]['average_embeddings'].to(device)
    target_cluster_average_embeddings = target_cluster_average_embeddings.unsqueeze(0)
    average_embeddings = torch.stack(df['average_embeddings'].tolist()).to(device)

    d = torch.cdist(target_cluster_average_embeddings, average_embeddings, p=2)
    #d = torch.cdist(target_word_embeddings_tensor, average_embeddings, p=2)
    # dの各行：1個ある （target_clusterの個数）
    # dの各列：average_embeddings個ある （全クラスターの個数）
    
    ## 最近傍 numberOfClusters個のクラスターを取得 （default=11 (自クラスタ含む)）
    sorted_d , sorted_indices = d.sort()
    nearest_cluster_indices = sorted_indices[:, :numberOfClusters][0]
    nearest_cluster_indices = nearest_cluster_indices.tolist()
    #nearest_cluster_indices.remove(plot_target_index)

    # nearest_cluster_indicesのdfを返す
    nearest_cluster_df = df.iloc[nearest_cluster_indices]
    return nearest_cluster_df



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
                    if type(df['notable_figer_types'][i]) == list:
                        category_dict[df['wiki_id'][i]] = df['notable_figer_types'][i][0]
                    else:
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
                    if type(df['notable_figer_types'][i]) == list:
                        category_dict[target_word] = df['notable_figer_types'][i][0]
                    else:
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
                if type(df['notable_figer_types'][i]) == list:
                    category_dict[target_word] = df['notable_figer_types'][i][0]
                else:
                    category_dict[target_word] = df['notable_figer_types'][i]
            if 'wiki_id'  in df.columns:
                wiki_id_dict[target_word] = df['wiki_id'][i]
            if 'target_word_sub_len'  in df.columns:
                target_word_sub_len_dict[target_word] = df['target_word_sub_len'][i]


        #target_word_embeddings_dict[target_word].append(torch.stack(target_word_embedding, dim=0))

    print('Creating sentence_count')
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
    print(f"Number of sentences： {aggregated_df['sentence_count'].sum()}")
    print(f"Number of target_word： {len(aggregated_df['target_word'])}\n")

    return aggregated_df




args = get_args()

## dataset install
df_list = multiple_read_jsonl(args.jsonl_path)
emb = multiple_load_tensor(args.emb_path)


## splitされたデータをconcatする
concat_df = pd.concat(df_list).reset_index(drop=True)

#if type(concat_df['notable_figer_types'][0]) is list :
#    concat_df['notable_figer_types'] = [types[0] for types in concat_df['notable_figer_types']]
#print(concat_df['notable_figer_types'])
print(f'len(concat_emb): {len(emb)}','\n')
#concat_df['notable_figer_types'] = [types[0] for types in concat_df['notable_figer_types']]
concat_df['target_word_embeddings_list'] = list(emb)
#concat_df['target_word_embeddings_list'] = torch.stack(emb, dim=0)
#print(concat_df.head())
#print(f"dataframe keys : {concat_df.keys()}")




print(f"output path : {args.output_path}")
if args.is_plot_nearest_cluster:
    if args.plot_target_word is None:
        print("Please set args:plot_target_word")
        exit()
    # dfを集約する
    aggregated_df = aggregate_df(concat_df, args.is_group_by_Wiki_id)
    #print(aggregated_df['target_word_embeddings_list'])

    # average_embeddingを作成
    average_embeddings_list = []
    print("Creating average Vector.")
    for embeddings_list in tqdm(aggregated_df['target_word_embeddings_list']):
        embeddings_tensor = torch.stack(embeddings_list, dim=0)
        average_embeddings_list.append(generate_average_vector(embeddings_tensor))
        #average_embeddings_list.append(generate_average_vector(embeddings_list))
    print("Done : generate_average_vector")
    aggregated_df['average_embeddings'] = average_embeddings_list

    nearest_cluster_df = get_nearest_cluster_df(aggregated_df, args.plot_target_word, numberOfClusters=args.numberOfClusters , is_group_by_Wiki_id=args.is_group_by_Wiki_id)

    classes = []
    if args.is_group_by_Wiki_id:
        for target_word_list , sentence_count in zip(nearest_cluster_df['target_word'], nearest_cluster_df['sentence_count']):
            classes.extend([target_word_list[0]]*sentence_count)
    else:
        for target_word, sentence_count in zip(nearest_cluster_df['target_word'], nearest_cluster_df['sentence_count']):
            classes.extend([target_word]*sentence_count)
    classes = np.array(classes)
    nearest_cluster_emb = torch.stack(list(itertools.chain.from_iterable(nearest_cluster_df['target_word_embeddings_list'])), dim=0)
    #category2marker = {'Non NE':'^', 'Person':'o', 'Location':'s'}
    #category2marker = make_category2marker(nearest_cluster_df)
    category2color = make_category2color(nearest_cluster_df, target_cluster_name=args.plot_target_word)
    #plot_embeddings(nearest_cluster_emb, outfile=args.output_path, emb_method="UMAP", classes=classes, class2marker=category2marker, s=args.marker_size)
    plot_embeddings(nearest_cluster_emb, outfile=args.output_path, emb_method="UMAP", classes=classes, class2color=category2color,  is_legend=False, s=args.marker_size)
    print(f"target_cluster color : {category2color[args.plot_target_word]}")
    
    #plot_embeddings(nearest_cluster_emb, outfile=args.output_path, emb_method="UMAP")
    
else:
    print("Creating category list for plot")
    super_category_list = []
    sub_category_list = []
    for notable_figer_types in tqdm(concat_df['notable_figer_types']):
        if pd.isna(notable_figer_types):
            super_category = 'Non NE' 
            sub_category = 'Non NE' 
        elif '/person' in notable_figer_types:
            super_category = 'Person'
            sub_category = notable_figer_types
        elif '/location' in notable_figer_types:
            super_category = 'Location'
            sub_category = notable_figer_types
        super_category_list.append(super_category)
        sub_category_list.append(sub_category)

    super_category_list = np.array(super_category_list)
    sub_category_list = np.array(sub_category_list)

    class2color = {'Non NE':'#ff7f0e', 'Person':'#1f77b4', 'Location':'#2ca02c'}
    plot_embeddings(emb, outfile=args.output_path, emb_method="UMAP", max_labels=len(emb), classes=super_category_list, class2color=class2color)

