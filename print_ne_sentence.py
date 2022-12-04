import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import os
import collections

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
#print(f"センテンス数： {'{:,}'.format(df['sentence_count'].sum())}")
#print(f"target_word種類数： {'{:,}'.format(len(df['target_word']))}")

#以下は集約されていない版
word_type_counter = collections.Counter(df['word_type'])
print(f"word_typeの分布： {word_type_counter}")
print(f"NEの数：{word_type_counter['ne']}，    NEの割合： {word_type_counter['ne']/len(df['word_type'])}")
print(f"NE以外の数：{word_type_counter['non_ne']}，    NE以外の割合： {word_type_counter['non_ne']/len(df['word_type'])}")
print()

print(df.head(args.n))



#for target_ne , notable_figer_types in zip(df['target_ne'], df['notable_figer_types']):
#    if 'person' in notable_figer_types[0] :
#        print(f'{target_ne} : {notable_figer_types}')

