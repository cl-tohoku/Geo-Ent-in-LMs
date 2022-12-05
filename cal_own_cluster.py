import argparse
import io,sys
import os
import csv
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import numpy.linalg as LA
import torch
from scipy.spatial.distance import cosine
from plot import plot_embeddings_bokeh
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from tqdm import tqdm
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True, nargs='*', 
                            help="input dataset path")
    parser.add_argument("--emb_path", required=True, nargs='*', 
                            help="")
    #parser.add_argument("--ave_emb_path", required=True, nargs='*', 
    #                        help="")
    parser.add_argument("--L_p", type=int, default=2,
                        help="Setting the L_p norm used in the distance function")
    parser.add_argument("--output_path", required=True, type=os.path.abspath, 
                            help="output csv path")
    parser.add_argument("--do_series", action='store_true',
                            help="")
    parser.add_argument("--do_parallel",  action='store_true',
                            help="")
    args = parser.parse_args()
    return args

def fn(i, average_embeddings, p, target_word_embeddings_list, sentence_count):
    try:
        pdist = torch.nn.PairwiseDistance(p=p)
        dist_list = []
        own_count = 0
        for target_word_embedding in target_word_embeddings_list:
            target_word_emb = torch.unsqueeze(target_word_embedding, 0)
            for j, ave_embedding in enumerate(average_embeddings):
                ave_emb = torch.unsqueeze(ave_embedding, 0)
                dist_list.extend(pdist(target_word_emb, ave_emb))
            if dist_list.index(min(dist_list)) == i:
                own_count += 1
            dist_list = []
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
    #cpu_num = os.cpu_count() // 2
    cpu_num = 1
    print(f'使用cpu数:{cpu_num}')
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
def series_cal_percentage_of_own_cluster(df, p=2):
    """
    input:
        dataframe
    
    Returns:
        own cluster count: List of int        
        own cluster percentage: List of floats 
    """

    print('直列処理')
    # distance function
    pdist = torch.nn.PairwiseDistance(p=p)
    own_count = 0
    own_count_list = []
    dist_list = []
    percentage_of_own_cluster = []
    for i, (target_word_embeddings_list, sentence_count) in enumerate(zip(tqdm(df['target_word_embeddings_list']), df['sentence_count'])):
        for target_word_embedding in target_word_embeddings_list:
            target_word_emb = torch.unsqueeze(target_word_embedding, 0)
            for j, ave_embedding in enumerate(df['average_embeddings']):
                ave_emb = torch.unsqueeze(ave_embedding, 0)
                dist_list.extend(pdist(target_word_emb, ave_emb))
            if dist_list.index(min(dist_list)) == i:
                own_count += 1
            dist_list = []

        #if args.category == 'ne':
        #    print(df['target_word'][i])
        #elif args.category == 'common_noun':
        #    print(df['noun'][i])
        #print(f'own_count = {own_count}')
        #print(f'sentence_count = {sentence_count}')
        #print(f'own_count/sentence_count = {own_count/sentence_count}\n')
        own_count_list.append(own_count)
        percentage_of_own_cluster.append(own_count/sentence_count)
        own_count = 0
    return own_count_list, percentage_of_own_cluster


def cal_micro_ave(list_1, list_2):
    return list_1.sum() / list_2.sum()

def save_df_to_csv(df, output_path):
    #TODO: result dir がなければ作成する
    print(f'savefile path: {output_path}')

    output_df = pd.DataFrame([df['target_word'], df['percentage_of_own_cluster'], df['own_count_list'],  df['sentence_count']]).T
    micro_ave = cal_micro_ave(df['own_count_list'], df['sentence_count'])
    print(f'micro_ave : {micro_ave}')
    output_df.append({"micro_ave" : micro_ave}, ignore_index=True)
    output_df.to_csv(output_path, encoding="utf_8_sig")

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

def aggregate_df(df):
    ## 処理用データ作成
    print('データを集約')
    sentence_dict = defaultdict(list)
    target_word_embeddings_dict = defaultdict(list)
    for target_word, sentence, target_word_embedding in zip(tqdm(df['target_word']), df['sentence'], df['target_word_embeddings_list']):
        sentence_dict[target_word].append(sentence)
        target_word_embeddings_dict[target_word].append(target_word_embedding)
        #target_word_embeddings_dict[target_word].append(torch.stack(target_word_embedding, dim=0))

    print('sentence_count 作成')
    sentence_count = [len(s) for s in list(sentence_dict.values())]

    ## _df作成
    aggregated_df = pd.DataFrame(
        data = {'target_word' : list(sentence_dict.keys()), 
                'sentence_list': list(sentence_dict.values()),
                'sentence_count': sentence_count,
                'target_word_embeddings_list' : list(target_word_embeddings_dict.values())
                }
        )
    print(aggregated_df.head())
    print(f"センテンス数： {aggregated_df['sentence_count'].sum()}")
    print(f"target_word数： {len(aggregated_df['target_word'])}\n")

    return aggregated_df

args = get_args()

## dataset install
df_list = multiple_read_jsonl(args.jsonl_path)
emb_list = multiple_load_tensor(args.emb_path)
#ave_emb_list = multiple_load_tensor(args.ave_emb_path)
#print_arg_path_list(args.jsonl_path)

## splitされたデータをconcatする
concat_df = pd.concat(df_list).reset_index(drop=True)
#print(f'len(concat_df): {len(concat_df)}','\n')
#print(f'len(concat_df[0]): {len(concat_df[0])}','\n')
#print(f'len(concat_df[0][0]): {len(concat_df[0][0])}','\n')

#concat_emb = list(chain.from_iterable(emb_list)) #extendなのでいらない
#concat_emb = emb_list
print(f'len(concat_emb): {len(emb_list)}','\n')

## ave_embは後で作る
#concat_ave_emb = list(chain.from_iterable(ave_emb_list))
#print(f'len(concat_ave_emb): {len(concat_ave_emb)}','\n')
#print(f"concat_df['sentence_count'].sum(): {concat_df['sentence_count'].sum()}","\n")


concat_df['target_word_embeddings_list'] = emb_list
#concat_df['average_embeddings'] = concat_ave_emb

# dfを集約する
aggregated_df = aggregate_df(concat_df)

# average_embeddingを作成
average_embeddings_list = []
for embeddings_list in aggregated_df['target_word_embeddings_list']:
    embeddings_tensor = torch.stack(embeddings_list, dim=0)
    average_embeddings_list.append(generate_average_vector(embeddings_tensor))
print("Done : generate_average_vector")
aggregated_df['average_embeddings'] = average_embeddings_list


if args.do_series:
    own_count_list , percentage_of_own_cluster = series_cal_percentage_of_own_cluster(aggregated_df, args.L_p)
elif args.do_parallel:
    own_count_list , percentage_of_own_cluster = parallel_cal_percentage_of_own_cluster(aggregated_df, args.L_p)
else:
    raise ("Error: There is no argument for do_series or do_parallel")
#result = cal_percentage_of_own_cluster(aggregated_df, args.L_p)
#print(result)

aggregated_df['own_count_list'] = own_count_list
aggregated_df['percentage_of_own_cluster'] = percentage_of_own_cluster

print(aggregated_df.head())

## dfのown clusterについて　csv形式で保存する
save_df_to_csv(aggregated_df, args.output_path)
