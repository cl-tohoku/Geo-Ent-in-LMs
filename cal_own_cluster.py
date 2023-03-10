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
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from tqdm import tqdm
import collections
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True, nargs='*', 
                            help="input dataset path")
    parser.add_argument("--emb_path", required=True, nargs='*', 
                            help="")
    parser.add_argument("--L_p", type=int, default=2,
                        help="Setting the L_p norm used in the distance function")
    parser.add_argument("--output_path", required=True, type=os.path.abspath, 
                            help="output csv path")
    parser.add_argument("--do_series", action='store_true',
                            help="")
    parser.add_argument("--do_parallel",  action='store_true',
                            help="")
    parser.add_argument("--is_group_by_Wiki_id", action='store_true',
                            help="")
    parser.add_argument("--cuda_number", type=str, default="0",
                        help="cuda number")  
    args = parser.parse_args()
    return args


def getTitleFromWikipediaURL(url):
    # 正規表現パターン
    if 'https' in url:
        pattern = r"https://en.wikipedia.org/wiki/(.+)"
    elif 'http' in url:
        if '/en.wikipedia.org/wiki/' in url:
            pattern = r"http://en.wikipedia.org/wiki/(.+)"
        elif '/en.wikipedia.org//wiki/' in url:
            pattern = r"http://en.wikipedia.org//wiki/(.+)"
        elif  '/en.wikipedia.org/' in url and '/' not in url.replace('http://en.wikipedia.org/', ''):
            pattern = r"http://en.wikipedia.org/(.+)"
        else :
            print("return None")
            return None
    # 正規表現オブジェクトを作成
    regex = re.compile(pattern)
    # URLをマッチさせる
    match = regex.match(url)
    # タイトルを取得
    try:
        title = match.group(1)
    except:
        return None
    title = title.replace('_', ' ')
    return title

def fn(i, average_embeddings, p, target_word_embeddings_list, sentence_count):
    try:
        
        #pdist = torch.nn.PairwiseDistance(p=p)
        #dist_list = []
        cdist = torch.cdist
        own_count = 0

        target_word_embeddings_tensor = torch.stack(target_word_embeddings_list)
        average_embeddings_tensor = torch.stack(average_embeddings.tolist())
        d = cdist(target_word_embeddings_tensor, average_embeddings_tensor, p=p)
        values,indices =  torch.min(d, dim=1)
        own_count = torch.count_nonzero(indices == i).item()

        #for target_word_embedding in target_word_embeddings_list:
        #    target_word_emb = torch.unsqueeze(target_word_embedding, 0)
        #    for j, ave_embedding in enumerate(average_embeddings):
        #        ave_emb = torch.unsqueeze(ave_embedding, 0)
        #        dist_list.extend(pdist(target_word_emb, ave_emb))
        #    if dist_list.index(min(dist_list)) == i:
        #        own_count += 1
        #    dist_list = []
    except ConnectionResetError:
        print('ConnectionResetError -> pass')
        pass
    except ConnectionRefusedError:
        print('ConnectionRefusedError -> pass')
        pass
    except EOFError:
        
        pass
    except socket.error as e:
        if e.errno != errno.ECONNRESET:
            raise # Not error we are looking for
        pass
    return own_count, own_count/sentence_count

## 各 Embeddingの最近傍のクラスタの中心が自クラスタである割合を算出
##  現状，並列で動かない
def parallel_cal_percentage_of_own_cluster(df, p=2):
    """
    Args:
        tokens_tensor (sbj): Torch tensor size [n_tokens]
            with token ids for each token in text
        or df？
    
    Returns:
        own cluster count: List of int        
        own cluster percentage: List of floats 
    """
    print('並列処理')
    # distance function
    #pdist = torch.nn.PairwiseDistance(p=p)
    #dist_list = []
    #own_count = 0
    #cpu_num = os.cpu_count() - 10
    cpu_num = 4
    print(f"使用cpu数:{cpu_num}")
    with tqdm(total=len(df)) as progress:
        try:
            with ProcessPoolExecutor(max_workers=cpu_num) as executor:  
            #with ThreadPoolExecutor(max_workers=cpu_num) as executor:  
                futures = []  # 処理結果を保存するlist
                for i, (target_word_embeddings_list, sentence_count) in enumerate(zip(df['target_word_embeddings_list'], df['sentence_count'])):
                    future = executor.submit(fn, i, df['average_embeddings'], p, target_word_embeddings_list, sentence_count)
                    future.add_done_callback(lambda p: progress.update()) 
                    futures.append(future)
                result = [f.result() for f in futures]
        except ConnectionResetError:
            print('ConnectionResetError -> pass')
            pass
        except ConnectionRefusedError:
            print('ConnectionRefusedError -> pass')
            pass
        except EOFError:
            print('EOFError -> pass')
            pass
        except socket.error as e:
            print('socket.error -> pass')
            pass
    

    #print(result)
    #for r in result:
    #    print(r)
    #own_count_list.append(own_count)
    #percentage_of_own_cluster.append(own_count/sentence_count)
    own_count_list = []
    percentage_of_own_cluster = []
    for r in result:
        own_count_list.append(r[0])
        percentage_of_own_cluster.append(r[1])
    print("Done")
    return own_count_list, percentage_of_own_cluster

# 直列ver
def series_cal_percentage_of_own_cluster(df, p=2, is_group_by_Wiki_id=False, cuda_number="0"):
    """
    input:
        dataframe
    
    Returns:
        own cluster count: List of int        
        own cluster percentage: List of floats 
    """

    ## device check
    device = torch.device(f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")

    print('直列処理')
    # distance function
    pdist = torch.nn.PairwiseDistance(p=p)
    own_count = 0
    own_count_list = []
    dist_list = []
    percentage_of_own_cluster = []
    wrong_cluster_list = []
    wrong_type_list = []
    wrong_pair_list = []
    for i, (target_word_embeddings_list, sentence_count) in enumerate(zip(tqdm(df['target_word_embeddings_list']), df['sentence_count'])):
        target_word_embeddings_tensor = torch.stack(target_word_embeddings_list).to(device)
        average_embeddings = torch.stack(df['average_embeddings'].tolist()).to(device)

        d = torch.cdist(target_word_embeddings_tensor, average_embeddings, p=2)
        values,indices =  torch.min(d, dim=1)
        own_count = torch.count_nonzero(indices == i).item()

        wrong_other_cluster_indices = indices[indices != i]
        wrong_own_clusterWord_indices = torch.where(indices != i)[0]
        if is_group_by_Wiki_id:
            wrong_cluster = [df['target_word'][k.item()][0] for k in wrong_other_cluster_indices]
            wrong_pair = [ df['target_word'][i][wrong_own_clusterWord_indices[j].item()] + ' : ' + df['target_word'][wrong_other_cluster_indices[j].item()][0] for j in range(len(wrong_other_cluster_indices))]
            wrong_type = [(df['notable_figer_types'][k.item()][0] if df['word_type'][k.item()]=='ne' else 'Non_NE') for k in wrong_other_cluster_indices]
            wrong_pair_list.append(collections.Counter(wrong_pair))
        else:
            wrong_cluster = [df['target_word'][k.item()] for k in wrong_other_cluster_indices]
            wrong_type = [(df['notable_figer_types'][k.item()] if df['word_type'][k.item()]=='ne' else 'Non_NE') for k in wrong_other_cluster_indices]
        wrong_cluster_list.append(collections.Counter(wrong_cluster))
        wrong_type_list.append(collections.Counter(wrong_type))

        #for target_word_embedding in target_word_embeddings_list:
        #    target_word_emb = torch.unsqueeze(target_word_embedding, 0)
        #    for j, ave_embedding in enumerate(df['average_embeddings']):
        #        ave_emb = torch.unsqueeze(ave_embedding, 0)
        #        dist_list.extend(pdist(target_word_emb, ave_emb))
        #    if dist_list.index(min(dist_list)) == i:
        #        own_count += 1
        #    dist_list = []

        own_count_list.append(own_count)
        percentage_of_own_cluster.append(own_count/sentence_count)
        own_count = 0
    return own_count_list, percentage_of_own_cluster, wrong_cluster_list, wrong_pair_list, wrong_type_list


def cal_micro_ave(list_1, list_2):
    return list_1.sum() / list_2.sum()

def save_df2jsonl(df, output_path):
    # save embeddings
    dirname = os.path.dirname(output_path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        
    new_df = df[['target_word', 'notable_figer_types', 'percentage_of_own_cluster', 'own_count_list',  'sentence_count', 'wrong_cluster', 'wrong_types', 'wiki_id', 'word_type']]

    if 'target_word_sub_len'  in df.columns:
        new_df['target_word_sub_len'] = df['target_word_sub_len']
    if 'alias_count'  in df.columns:
        new_df['alias_count'] = df['alias_count']
    if 'wrong_pair'  in df.columns:
        new_df['wrong_pair'] = df['wrong_pair']

    # df →　jsonl形式で保存する
    print(f'savefile path: {output_path}')
    new_df.to_json(output_path, orient='records', force_ascii=False, lines=True)



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
    torch_list = []
    # listではなくはじめからtensorでする?
    for path in path_list:
        print(f'loading : {path}')
        #torch_list.append(torch.load(path))
        torch_list.extend(torch.load(path))
    print()
    return torch_list

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
emb_list = multiple_load_tensor(args.emb_path)


## splitされたデータをconcatする
concat_df = pd.concat(df_list).reset_index(drop=True)


print(f'len(concat_emb): {len(emb_list)}','\n')
print(f'len(concat_df): {len(concat_df)}','\n')


concat_df['target_word_embeddings_list'] = emb_list

# dfを集約する
aggregated_df = aggregate_df(concat_df, args.is_group_by_Wiki_id)

# average_embeddingを作成
average_embeddings_list = []
for embeddings_list in aggregated_df['target_word_embeddings_list']:
    embeddings_tensor = torch.stack(embeddings_list, dim=0)
    average_embeddings_list.append(generate_average_vector(embeddings_tensor))
print("Done : generate_average_vector")
aggregated_df['average_embeddings'] = average_embeddings_list

# WikipediaID → wiki_title
#print("WikipediaID から wiki_title を作成")
#wiki_title = [getTitleFromWikipediaURL(wiki_id) for wiki_id in aggregate_df['wiki_id']]
#print(f"len(wiki_title) : {len(wiki_title)}")
#print(f"len(aggregate_df) : {len(aggregate_df)}")
#aggregated_df['wiki_title'] = wiki_title 


if args.do_series:
    own_count_list , percentage_of_own_cluster, wrong_cluster_list, wrong_pair_list, wrong_type_list = series_cal_percentage_of_own_cluster(aggregated_df, args.L_p, args.is_group_by_Wiki_id, args.cuda_number)
elif args.do_parallel:
    own_count_list , percentage_of_own_cluster = parallel_cal_percentage_of_own_cluster(aggregated_df, args.L_p)
else:
    raise ("Error: There is no argument for do_series or do_parallel")
#result = cal_percentage_of_own_cluster(aggregated_df, args.L_p)
#print(result)

aggregated_df['own_count_list'] = own_count_list
aggregated_df['percentage_of_own_cluster'] = percentage_of_own_cluster
aggregated_df['wrong_cluster'] = wrong_cluster_list
aggregated_df['wrong_types'] = wrong_type_list
if wrong_pair_list != []:
    aggregated_df['wrong_pair'] = wrong_pair_list

print(aggregated_df.head())

## dfのown clusterについて　csv形式で保存する
save_df2jsonl(aggregated_df, args.output_path)

# dfも保存するとよさそう？
