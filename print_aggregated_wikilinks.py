import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import os
import collections

pd.set_option('display.max_rows', 100)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=os.path.abspath, 
                        help="input dataset path")
    parser.add_argument("--n", type=int, default=5, 
                        help="print num")
    args = parser.parse_args()
    return args

def read_json(path):
    print('json reading')
    return pd.read_json(path, orient='records', lines=True)

args = get_args()
print(args.input)
df = read_json(args.input)

#以下は集約された版
print(f"センテンス数： {'{:,}'.format(df['sentence_count'].sum())}")
print(f"target_word種類数： {'{:,}'.format(len(df['target_word']))}")
print(df.head(args.n))
print(df.columns)

#以下は集約されていない版
#word_type_counter = collections.Counter(df['word_type'])
#print(f"word_typeの分布： {word_type_counter}")
#print(f"NEの数：{word_type_counter['ne']}，    NEの割合： {word_type_counter['ne']/len(df['word_type'])} = {word_type_counter['ne']/len(df['word_type'])* 100}%")
##print(f"NE以外の数：{word_type_counter['non_ne']}，    NE以外の割合： {word_type_counter['non_ne']/len(df['word_type'])} = {word_type_counter['non_ne']/len(df['word_type']) * 100}%")
#print(f"common_nounの数：{word_type_counter['common_noun']}，    common_nounの割合： {word_type_counter['common_noun']/len(df['word_type'])} = {word_type_counter['common_noun']/len(df['word_type']) * 100}%")
#print()


#print(df.head(args.n))
#print(df[['target_word', 'notable_figer_types', 'wiki_id']].head(args.n))
#print()
#print(df[df['target_word'] == 'child'][['target_word', 'notable_figer_types', 'wiki_id']])
#print(df['notable_figer_types'][0])
#print(type(df['notable_figer_types'][0]))

#print( df['/person' in df['notable_figer_types'] ][['target_word', 'notable_figer_types', 'wiki_id']])
#for target_word, notable_figer_types, wiki_id, sentence_list in zip(df['target_word'], df['notable_figer_types'], df['wiki_id'], df['sentence_list']):
#    if 'building' in notable_figer_types[0]:
#        print(f"{target_word}\t{notable_figer_types}\t{wiki_id}\t{sentence_list[0]}\n")

#for target_ne , notable_figer_types in zip(df['target_word'], df['notable_figer_types']):
#    if 'person' in notable_figer_types[0] :
#        print(f'{target_ne} : {notable_figer_types}')

