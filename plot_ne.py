import pandas as pd
import matplotlib.pyplot as plt
import requests
import copy
import os
from collections import defaultdict
import numpy as np
import torch
import csv
from tqdm import tqdm
import seaborn as sns
from matplotlib import ticker
import random
import japanize_matplotlib
import argparse
import collections


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=os.path.abspath, 
                        help="input path")
    parser.add_argument("--output", type=os.path.abspath, 
                        help="output path")
    args = parser.parse_args()
    return args

def print_name_location_ratio(ne_df):
    print("人名と地名の割合")
    print(f"len(ne_df) : {len(ne_df)}")
    print(f"len(人名) / len(ne_df) : {len(ne_df[ne_df['notable_figer_types'].str.contains('/person/')]) / len(ne_df)}")
    print(f"len(地名) / len(ne_df) : {len(ne_df[ne_df['notable_figer_types'].str.contains('/location/')]) / len(ne_df)}")


def plot_subword_len_ratio(ne_df, ne_subword_len_fig_path, personAndLocation_len_fig_path):
    print("\nサブワードの割合をプロット")
    sns.set()
    plt.figure() 
    sns.distplot(ne_df['target_word_sub_len'], kde=False, label='NE (Person and Location)')
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #plt.xlim(1, max(ne_df['target_word_sub_len']))
    plt.legend()
    plt.savefig(ne_subword_len_fig_path) # ne_subword_len_fig_path = "./result/ne_subword_len.png"
    plt.show()
    plt.figure()
    ax = plt.gca()
    sns.distplot(ne_df[ne_df['notable_figer_types'].str.contains('/person/')]['target_word_sub_len'], kde=False, label='Person')
    sns.distplot(ne_df[ne_df['notable_figer_types'].str.contains('/location/')]['target_word_sub_len'], kde=False, label='Location')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #plt.xlim(1, max(ne_df['target_word_sub_len']))
    plt.legend()
    plt.savefig(personAndLocation_len_fig_path) #personAndLocation_len_fig_path = "./result/personAndLocation_subword_len.png"
    plt.show()

def plot_hist(args_list, title="", xlabel="", ylabel="", output_path="./result/plot_hist.png", xlim=None):
    sns.set()
    sns.set_style('whitegrid')
    #japanize_matplotlib.japanize()
    x = args_list
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if xlim is not None:
        plt.xlim(min(args_list)-1, xlim)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.hist(x, color='b')
    sns.distplot(x, kde=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(f"saving... : {output_path}")
    plt.savefig(output_path)
    plt.show()
    print("Done")

args = get_args()
print("data loading...")
print(args.input)
#input_path = '/work/masaki/data/wikilinks/ne/Person_Location/separete/target_word_in_sentence_replaced_wikilinks_more10_title.jsonl'
ne_df = pd.read_json(args.input, orient='records', lines=True)


alias_count = [len(set(target_word)) for target_word in ne_df['target_word']]
print(f"max : {max(alias_count)}")
print(f"len : {len(alias_count)}")

c = collections.Counter(alias_count)
print(c)
sorted_c = sorted(c.items(), key = lambda d : d[1], reverse=True)
#plot_hist(alias_count, output_path="./result/target_word_hist.png", xlim=20)

print(len(ne_df[((ne_df['alias_count'] >= 2) & (ne_df['alias_count'] <= 9))]))

# alias_countを含むdfを保存
#output = "/work/masaki/data/wikilinks/ne/Person_Location/separete/target_word_in_sentence_replaced_wikilinks_more10_title_sub1_5.jsonl"
#print(f"saving : {args.output}")
#ne_df["alias_count"] =  alias_count
#ne_df.to_json(args.output, orient='records', force_ascii=False, lines=True)
#print("Done")



#c = collections.Counter(alias_count)
#print(c)
#sorted_c = sorted(c.items(), key = lambda d : d[1], reverse=True)
#plot_hist(alias_count, output_path="./result/target_word_hist.png", xlim=20)
#
#SENTENCE_COUNT_UPPER = 16056
#sample_sentence_num = 3
#sentence_sum = 0
#for ne_alias, count in sorted_c:
#    if ne_alias == 1 or ne_alias == 2:
#        continue
#    if  SENTENCE_COUNT_UPPER <= sentence_sum:
#        print(f"ne_alias : {ne_alias}")
#        print(f"count : {count}")
#        break 
#    sentence_sum += ne_alias * sample_sentence_num * count
#print(f"sentence_sum : {sentence_sum}")

# aliasの数を nからk の範囲を決めたい (カバレッジを見てみる)
# aliasセンテンスの全体の-割を占めるデータを取得
#SENTENCE_COUNT_UPPER = (len(alias_count)-c[1])*0.8
#print(f"SENTENCE_COUNT_UPPER : {SENTENCE_COUNT_UPPER}")
#print("\n2～")
#sentence_sum = 0
#for ne_alias, count in sorted_c:
#    if ne_alias <= 1  :
#        continue
#    if  (len(alias_count)-c[1])*0.9 <= sentence_sum:
#        print(f"ne_alias : {ne_alias}")
#        print(f"count : {count}")
#        break 
#    sentence_sum += count
#print(f"sentence_sum : {sentence_sum}")
#print(f"(len(alias_count)-c[1])*0.9 : {(len(alias_count)-c[1])*0.9}")
#print(f"sentence_sum / (len(alias_count)-c[1]) : {sentence_sum / (len(alias_count)-c[1])}")
#
#print("\n3～")
#sentence_sum = 0
#for ne_alias, count in sorted_c:
#    if ne_alias <= 2  :
#        continue
#    if  (len(alias_count)-c[1])*0.9 <= sentence_sum:
#        print(f"ne_alias : {ne_alias}")
#        print(f"count : {count}")
#        break 
#    sentence_sum += count
#print(f"sentence_sum : {sentence_sum}")
#print(f"(len(alias_count)-c[1])*0.9 : {(len(alias_count)-c[1])*0.9}")
#print(f"sentence_sum / (len(alias_count)-c[1]) : {sentence_sum / (len(alias_count)-c[1])}")





#print(f"全データの90% ： {len(ne_df)*0.9}")
## 全データの9割 ： 3251857.5
#print(f"全データの95%  ： {len(ne_df)*0.95}")
## 全データの95%  ： 3432516.25

#print(f"サブワードが1～6のもののみ抽出した場合 ： {len(ne_df[ne_df['target_word_sub_len'] <= 6])}")
### サブワードが1～6のもののみ抽出した場合 ： 3553040
#
#print(f"サブワードが1～6のデータの網羅率 ： {len(ne_df[ne_df['target_word_sub_len'] <= 6])/len(ne_df)}")
### サブワードが1～6のデータの網羅率 ： 0.983356743030714


#print(f"サブワードが1～5のもののみ抽出した場合 ： {len(ne_df[ne_df['target_word_sub_len'] <= 5])}")
## サブワードが1～5のもののみ抽出した場合 ： 3443671

#print(f"サブワードが1～5のデータの網羅率 ： {len(ne_df[ne_df['target_word_sub_len'] <= 5])/len(ne_df)}") # これを採用する
## サブワードが1～5のデータの網羅率 ： 0.9530872432140707

#print(f"サブワードが1～4のもののみ抽出した場合 ： {len(ne_df[ne_df['target_word_sub_len'] <= 4])}")
### サブワードが1～4のもののみ抽出した場合 ： 3198953
#
#print(f"サブワードが1～4のデータの網羅率 ： {len(ne_df[ne_df['target_word_sub_len'] <= 4])/len(ne_df)}")
### サブワードが1～4のデータの網羅率 ： 0.8853578916050289


#print("サブワード長を1～5に制限，人名と地名の割合")
#sub_ne_df = ne_df[ne_df['target_word_sub_len'] <= 5]
#print(f"len(sub_ne_df) : {len(sub_ne_df)}")
#print(f"len(人名) / len(sub_ne_df) : {len(sub_ne_df[sub_ne_df['notable_figer_types'].str.contains('/person/')]) / len(sub_ne_df)}")
#print(f"len(地名) / len(sub_ne_df) : {len(sub_ne_df[sub_ne_df['notable_figer_types'].str.contains('/location/')]) / len(sub_ne_df)}")


# サブワード長を1～5に制限したdataframeを保存
#output = "/work/masaki/data/wikilinks/ne/Person_Location/separete/target_word_in_sentence_replaced_wikilinks_more10_title_sub1_5.jsonl"
#print(f"saving : {args.output}")
#sub_ne_df.to_json(args.output, orient='records', force_ascii=False, lines=True)
#print("Done")


#sub_ne_df_index = list(sub_ne_df.index)


## サブワード長を1～5に制限したtensorを抽出
#tensor_path = "/work/masaki/data/wikilinks/ne/Person_Location/separete/target_word_in_sentence_replaced_wikilinks_more10_title_tensor.pt"
#tensor = torch.load(args.tensor_path)
# indexのリストを使用してtensorから要素を取得
#selected_tensor = tensor[sub_ne_df_index]
# 取得した要素を表示
#print(selected_tensor.size())  
# save 
#output_torch_file = "/work/masaki/data/wikilinks/ne/Person_Location/separete/target_word_in_sentence_replaced_wikilinks_more10_title_tensor_sub1_5.pt"
#torch.save(selected_tensor, output_torch_file )


## サブワード長を1～5に制限したtensorをランダムサンプルする
#SEED = 42
#random.seed(SEED)
#sample_size = 15962
#sampled_indices = random.sample(range(selected_tensor.shape[0]), sample_size)
#sampled_tensor = selected_tensor[sampled_indices, :] #これは前から取得してるっぽい
# save 
#output_sampled_tensor_file = "/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/target_word_in_sentence_replaced_wikilinks_more10_title_tensor_sub1_5_15962.pt"
#torch.save(sampled_tensor, output_sampled_tensor_file )
#print(sampled_tensor.size())



## サブワード長を1～5に制限したdataframeをランダムサンプルする
# ランダムサンプルするindex -> sampled_indices = random.sample(range(selected_tensor.shape[0]), sample_size)
#sampled_sub_ne_df = sub_ne_df.iloc[sampled_indices] #loc[]ではエラーとなる
#sampled_sub_ne_df_output_path = "/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/sampled_15962/wikilinks_ne.jsonl"
#print(f"saving : {sampled_sub_ne_df_output_path}")
#sampled_sub_ne_df.to_json(sampled_sub_ne_df_output_path, orient='records', force_ascii=False, lines=True)
#print("Done")

#print_name_location_ratio(sampled_sub_ne_df)

#plot_subword_len_ratio(sampled_sub_ne_df, ne_subword_len_fig_path="./result/sampled_ne_subword_len.png", personAndLocation_len_fig_path="./result/sampled_personAndLocation_subword_len.png")