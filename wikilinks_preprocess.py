import pandas as pd
import matplotlib.pyplot as plt
import collections
import requests
import copy
from collections import defaultdict
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import argparse
import io,sys
import os
import csv
from tqdm import tqdm
import nltk
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import spacy
from spacy.symbols import NOUN
import io
import pickle
import random

model = BertModel.from_pretrained('bert-base-uncased',
           output_hidden_states = True,)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MyDataset(Dataset):
    def __init__(self, path):
        #self.csv_df = pd.read_csv(path)
        self.df = pd.read_json(path, orient="records", lines=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target_word = self.df['target_word'][idx]
        sentence =  self.df['sentence'][idx]
        word_type = self.df['word_type'][idx]
        return target_word, sentence, word_type


"""
今後基本的には，以下のデータ形式をモデルに入力させる (以前の形式とは異なる)
wikilinks_df = pd.DataFrame(
        data = { 'target_word' : target_word_list, 
                 'sentence': sentence_list
                }
        )
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=os.path.abspath, 
                        help="input dataset path")
    parser.add_argument("--input_list", nargs='*',  
                        help="input dataset path list")
    parser.add_argument("--output", type=os.path.abspath, 
                        help="save dataset path")
    parser.add_argument('--sentence_count_lower', type=int, default=None,
                        help='Lower limit on the number of sentences')
    parser.add_argument("--split", type=int, default=0,
                        help="data split")
    parser.add_argument("--before_name", type=str, default=None,
                        help="")
    parser.add_argument("--after_name", type=str, default=None,
                        help="")
    parser.add_argument("--preprocessors", type=str, nargs='+',
                        choices=['extract_df_frequency_more_X',
                                 'data_formating_for_df',
                                 'delete_sentence_512tokens_over' ,
                                 'save_split_jsonl',
                                 'rename_df_columuns',
                                 'create_target_word_in_sentence_and_512token_less',
                                 'create_common_noun_vocab_df',
                                 'create_aggregation_df',
                                 'create_aggregation_df_more_k',
                                 'create_separete_aggregattion_df',
                                 'create_df_ne_sentence_same_length',
                                 'create_df_target_word_same_length',
                                 'save_split_jsonl',
                                 'add_word_type_column',
                                 'create_concat_tensor',
                                 'create_concat_column',
                                 'create_delete_symbol_df',
                                 'create_mix_data' ],
                        help="")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="")
    parser.add_argument("--is_vocab",  action='store_true',
                        help="")
    parser.add_argument("--is_human",  action='store_true',
                        help="")
    parser.add_argument("--word_type",  type=str,
                        help="")          
    parser.add_argument("--target_word_path_list", nargs='*', 
                        help="")
    parser.add_argument("--sentence_path_list", nargs='*', 
                        help="") 
    parser.add_argument("--word_type_path_list", nargs='*', 
                        help="")      
    parser.add_argument("--SEED", type=int, default=42,
                        help="")      
    parser.add_argument("--MAX_SENTENCE_NUM", type=int, 
                        help="") 
    args = parser.parse_args()
    return args

def read_json(path):
    print(f'json reading : {path}')
    try:
        df = pd.read_json(path, orient='records', lines=True)
    except Exception as error:
        print(f'error path : {path}')
        raise error
    return df

def multiple_read_jsonl(path_list):
    df_list = []
    for path in path_list:
        df_list.append(read_json(path))
    return df_list

def read_txt(path):
    """
    output:list
    """
    with open(path, 'r') as f:
        txt_list = f.read().split("\n")
    return txt_list

def multiple_load_tensor(path_list):
    print("multiple_load_tensor")
    tensor_list = []
    for path in path_list:
        print(f'loading : {path}')
        tensor = torch.load(path)
        #tensor = np.ravel(tensor_array) #平坦化
        ##tensor = np.ravel(tensor)
        #x = torch.from_numpy(tensor.astype(np.float32)).clone()
        #tensor = torch.stack(x, dim=0)
        tensor_list.extend(tensor)
    #print(f'len(tensor_list) :{len(tensor_list)}')
    #print(f'len(tensor_list[0]) :{len(tensor_list[0])}')
    #raise 
    return tensor_list

# wikilinksのX以上のセンテンスのインスタンスのみを新たなdfとして保存する
def extract_df_frequency_more_X(path, args):
    X = args.sentence_count_lower
    wiki_df = read_json(path)
    mentions = wiki_df['mention']

    print('mention list 作成')
    mention_set_list = list(set(mentions))

    print('count mentions')
    cnt = collections.Counter(mentions)

    sorted_cnt = sorted(cnt.items(), key=lambda x:x[1], reverse=True)
    mention_set_list_more_X = [i for i in mention_set_list if cnt[i] >= X]

    print(f'len(sorted_cnt) : {len(sorted_cnt)}')
    print(f'len(mention_set_list_more_X) : {len(mention_set_list_more_X)}')

    wiki_df_more_X = wiki_df.query('mention in @mention_set_list_more_X')
    print(wiki_df_more_X.head())

    # df →　jsonl形式で保存する
    wiki_df_more_X.to_json('/data/wikilinks/wikilinks_more_'+str(X)+'.jsonl', orient='records', force_ascii=False, lines=True)

#データを整形する (以前の形式)
def data_formating_for_df(path, args):
    #path = '/data/wikilinks/wikilinks_more_'+str(X)+'.jsonl'
    wiki_df = read_json(path)
    print('mention set list 作成')
    mention_set_list = list(set(wiki_df['mention']))

    ## 処理用データ作成
    print('処理用データ作成')
    sentence_dict = defaultdict(list)
    #tokenized_sentence_dict = defaultdict(list)
    #sentence_tokens_tensor_dict = defaultdict(list)
    #segments_tensors_dict = defaultdict(list)
    notable_figer_types_dict = {}
    wiki_id_dict = {}

    for left_ctx, mention, right_ctx, notable_figer_types, wiki_id in zip(tqdm(wiki_df['left_ctx']), wiki_df['mention'], wiki_df['right_ctx'] , wiki_df['notable_figer_types'] , wiki_df['wiki_id'] ):
        sentence = left_ctx + " " +mention + " " +right_ctx
        #tokenized_sentence, sentence_tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
        sentence_dict[mention].append(sentence)
        #tokenized_sentence_dict[mention].append(tokenized_sentence)
        #sentence_tokens_tensor_dict[mention].append(sentence_tokens_tensor)
        #segments_tensors_dict[mention].append(segments_tensors)
        notable_figer_types_dict[mention] = notable_figer_types
        wiki_id_dict[mention] = wiki_id

    print('sentence_count 作成')
    sentence_count = [len(s) for s in list(sentence_dict.values())]

    ## wikilinks_df作成
    wikilinks_df = pd.DataFrame(
        data = { 'target_word' : list(sentence_dict.keys()), 
                'sentence_list': list(sentence_dict.values()),
                'sentence_count': sentence_count,
                'wiki_id': list(wiki_id_dict.values()), 
                'notable_figer_types': list(notable_figer_types_dict.values())
                }
        )
    print(wikilinks_df.head())
    print(f"センテンス数： {wikilinks_df['sentence_count'].sum()}")
    print(f"NE数： {len(wikilinks_df['target_word'])}")

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    dir_name = os.path.dirname(path)
    wikilinks_df.to_json(dir_name + '/preprocessed_'+ basename_without_ext +'.jsonl', orient='records', force_ascii=False, lines=True)


## 前処理済みデータから512トークン以上のセンテンスを削除する (以前の形式)
def delete_sentence_512tokens_over(path):
    print(f'delete_sentence_512tokens_over')
    
    df = read_json(path)
    print("Computing... ")
    for i, (sentence_list, sentence_count) in enumerate(zip(tqdm(df['sentence_list']), df['sentence_count'])):
        delete_frag = False
        for j, sentence in enumerate(sentence_list):
            tokenized_sentence, sentence_tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
            if is_512tokens_over(tokenized_sentence):
                print('Token count is higher than 512. -> delete sentence')
                delete_frag = True
                del sentence_list[j]
                sentence_count = sentence_count -1
        if delete_frag:
            df['sentence_list'][i] = sentence_list
            df['sentence_count'][i] = sentence_count
    return df
        

def is_512tokens_over(sentence):
    if len(sentence) > 512:
        return True
    else :
        return False

## (target_word in sentence) のデータを抽出して保存する && バッチ単位で処理 (sentenceが512トークン以下) 
def create_target_word_in_sentence_and_512token_less(path, args):
    print("create_target_word_in_sentence_and_512token_less")
    
    batch_size = args.batch_size
    print(f"batch_size: {batch_size}")
    sentence_dataset = MyDataset(path)
    dataloader = DataLoader(sentence_dataset, batch_size=batch_size)

    # 最大入力長
    INPUT_MAX_LENGTH = 512
    target_word_embeddings_list = []
    target_word_list = []
    sentence_list = []
    word_type_list = []
    for i, data in enumerate(tqdm(dataloader)):
        target_word, sentence, word_type = data
        tokenized_target_word = tokenizer(target_word, add_special_tokens=False)
        tokenized_sentence = tokenizer(sentence, max_length=INPUT_MAX_LENGTH, padding=True, truncation=True, return_length=True)
        #tokenized_sentence = tokenizer(sentence, padding=True,  return_length=True)
        for j, (tokenized_sentence_len, tokenized_sentence_ids, tokenized_target_word_ids) in enumerate( zip(tokenized_sentence['length'], tokenized_sentence['input_ids'], tokenized_target_word['input_ids'])):
            ## 512token以内 and sentence中にtarget_wordが含まれるデータのみ抽出 (かつ，現状はサブワードの1tokenのみとしている)
            ## (tokenized_sentence_len <= INPUT_MAX_LENGTH)は別にいらない (max_length=INPUT_MAX_LENGTHとしているので)
            if (tokenized_sentence_len <= INPUT_MAX_LENGTH) and (tokenized_target_word_ids[0] in tokenized_sentence_ids) and (len(tokenized_target_word_ids) == 1):
              target_word_list.append(target_word[j])
              sentence_list.append(sentence[j])
              word_type_list.append(word_type[j])

            else:
                print(f"{j}: センテンスが{INPUT_MAX_LENGTH}トークン以上 or センテンス中にtarget_wordが含まれていません or len(tokenized_target_word_ids)が2以上です")
                print('skipします')


    new_df = pd.DataFrame(
        data = {'target_word' : target_word_list, 
                'sentence': sentence_list,
                'word_type' : word_type_list
                }
        )
    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    #basename_without_ext = basename_without_ext.replace('preprocessed_', '')
    dir_name = os.path.dirname(path)
    print("save")
    new_df.to_json(dir_name + '/target_word_in_sentence_'+ basename_without_ext +'.jsonl', orient='records', force_ascii=False, lines=True)


def get_split_df_index(df, k):
    n = df.shape[0]
    return (np.array_split([i for i in range(0, n)], k))

def df_shuffle(df):
    """
    shuffle df raw
    input: df
    output: shuffled df
    """
    seed = 42
    return df.sample(frac=1, random_state=seed, ignore_index=True)

def save_split_jsonl(split_len, input_jsonl_path=None, input_df=None, output_path=None, is_shuffle=True):
    k = split_len
    print('save split jonsl')
    if input_jsonl_path is not None:
        df = read_json(input_jsonl_path)
    elif input_df is not None:
        df = input_df
    else:
        raise 

    if is_shuffle:
        df = df_shuffle(df)

    if output_path is not None:
        basename = output_path.replace('.jsonl', '')
    else:
        basename = input_jsonl_path.replace('.jsonl', '')

    # TODO: edit file name
    split_index = get_split_df_index(df,k)
    for i, index in enumerate(split_index):
        start = index[0]
        end = index[-1] + 1
        split_df = df.iloc[start:end, :]
        split_df.to_json(basename + '_split'+str(i+1)+'.jsonl', orient='records', force_ascii=False, lines=True)
        print('saved : '+ basename + '_split'+str(i+1)+'.jsonl')
    print('Done')


def rename_df_columuns(path, args):
    before_name = args.before_name
    after_name = args.after_name
    #path = '/data/wikilinks/preprocessed_wikilinks_more_'+str(args.sentence_count_lower)+'.jsonl'
    
    df = read_json(path)
    if before_name in df.columns :
        print(df.head())
        new_df = df.rename(columns={before_name: after_name})
        print(new_df.head())
        print('save')
        new_df.to_json(path, orient='records', force_ascii=False, lines=True)


# 整形済みデータから人名のみ取得→create_ne_dfに統合
#def extract_df_human_and_save(path):
#    print('extract_person_name')
#    
#    df = read_json(path)
#    basename = path.replace('.jsonl', '')
#
#    ## 処理用データ作成
#    print('処理用データ作成')
#    extract_target_ne = []
#    extract_sentence_list = []
#    extract_sentence_count = []
#    extract_wiki_id = []
#    extract_notable_figer_types = []
#
#    for target_ne, sentence_list, sentence_count, wiki_id, notable_figer_types in zip( tqdm(df['target_word']), df['sentence_list'], df['sentence_count'], df['wiki_id'], df['notable_figer_types']):
#        if 'person' in notable_figer_types[0] :
#            extract_target_ne.append(target_ne)
#            extract_sentence_list.append(sentence_list)
#            extract_sentence_count.append(sentence_count)
#            extract_wiki_id.append(wiki_id)
#            extract_notable_figer_types.append(notable_figer_types)
#
#    ## wikilinks_df作成
#    human_wikilinks_df = pd.DataFrame(
#        data = { 'target_word' : extract_target_ne, 
#                'sentence_list': extract_sentence_list,
#                'sentence_count': extract_sentence_count,
#                'wiki_id': extract_wiki_id, 
#                'notable_figer_types': extract_notable_figer_types
#                }
#        )
#    print(human_wikilinks_df.head())
#
#    # df →　jsonl形式で保存する
#    human_wikilinks_df.to_json(basename +'_human.jsonl', orient='records', force_ascii=False, lines=True)


#TODO:  前処理済みdfからBERTs の語彙との共通部分  OR  人名であるものを抽出して保存する
#TODO: data_formating_for_dfと統合する → 処理が遅いので，extract_df_human_and_saveっぽくする．後にextract_df_human_and_saveを統合する → argsでコントロールする
# target_wordの集約されている (以前の形式)
def create_ne_df(path, args, tokenizer=tokenizer):
    ## CutTokenじゃないので注意
    #path = '/data/wikilinks/preprocessed_wikilinks_more_'+str(args.sentence_count_lower)+'.jsonl'
    if args.is_vocab == False and args.is_human == False:
        raise ValueError("is_vocab and is_human are False")
    
    df = read_json(path)
    basename = path.replace('.jsonl', '')
    
    if args.is_vocab:
        vocab_list = list(tokenizer.get_vocab().keys())
    
    print(df.head())

    ## 処理用データ作成
    print('処理用データ作成')
    extract_target_ne = []
    extract_sentence_list = []
    extract_sentence_count = []
    extract_wiki_id = []
    extract_notable_figer_types = []

    for target_ne, sentence_list, sentence_count, wiki_id, notable_figer_types in zip( tqdm(df['target_word']), df['sentence_list'], df['sentence_count'], df['wiki_id'], df['notable_figer_types']):
        if (args.is_vocab == True) and (target_ne in vocab_list):
            extract_target_ne.append(target_ne)
            extract_sentence_list.append(sentence_list)
            extract_sentence_count.append(sentence_count)
            extract_wiki_id.append(wiki_id)
            extract_notable_figer_types.append(notable_figer_types)
        elif (args.is_person == True) and ('person' in notable_figer_types[0]):
            extract_target_ne.append(target_ne)
            extract_sentence_list.append(sentence_list)
            extract_sentence_count.append(sentence_count)
            extract_wiki_id.append(wiki_id)
            extract_notable_figer_types.append(notable_figer_types)
        

    ## wikilinks_df作成
    wikilinks_df = pd.DataFrame(
        data = {'target_word' : extract_target_ne, 
                'sentence_list': extract_sentence_list,
                'sentence_count': extract_sentence_count,
                'wiki_id': extract_wiki_id, 
                'notable_figer_types': extract_notable_figer_types
                }
        )
    
    print(f"センテンス数： {wikilinks_df['sentence_count'].sum()}")
    print(f"NE数： {len(wikilinks_df['target_word'])}")
    print(wikilinks_df.head())
    # df →　jsonl形式で保存する
    if args.is_vocab:
        wikilinks_df.to_json(basename +'_vocab.jsonl', orient='records', force_ascii=False, lines=True)
    elif args.is_human:
        wikilinks_df.to_json(basename +'_human.jsonl', orient='records', force_ascii=False, lines=True)



# 前処理していないwikilinks_more10.jsonlを対象にNE 以外の語彙のみでデータを作成する
# 出力データはtarget_wordの集約はしていない
def create_non_ne_vocab_df(path, args, tokenizer=tokenizer):
    nltk.download('punkt')
    print(f'前処理していない{path}を対象にNE以外の語彙のみでデータを作成する')
    #path = '/data/wikilinks/wikilinks_more_'+str(X)+'.jsonl'
    df = read_json(path)
    mention_set_list = list(set(df['mention']))
    vocab_list = list(tokenizer.get_vocab().keys())

    ## 処理用データ作成
    print('処理用データ作成')
    target_word_list = []
    sentence_list = []
    sentence_dict = defaultdict(list)

    for left_ctx, mention, right_ctx in zip(tqdm(df['left_ctx']), df['mention'], df['right_ctx']):
        sentence = left_ctx + " " +mention + " " +right_ctx
        tokenized_sentence = nltk.word_tokenize(sentence)
        ## TODO: nltkは後にSpacyに置き換える
        for token in tokenized_sentence:
            if ((token in vocab_list) and (token not in mention_set_list))  and (sentence not in sentence_dict[token]): #if (vocabに含まれる単語 and NE以外) and sentenceの重複なし
                sentence_dict[token].append(sentence)
                sentence_list.append(sentence)
                target_word_list.append(token)


    ## wikilinks_df作成
    wikilinks_df = pd.DataFrame(
        data = { 'target_word' : target_word_list, 
                 'sentence': sentence_list
                }
        )
    print(wikilinks_df.head())

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    dir_name = os.path.dirname(path)
    wikilinks_df.to_json(dir_name + '/preprocessed_non_ne_vocab_'+ basename_without_ext +'.jsonl', orient='records', force_ascii=False, lines=True)


# target_wordの集約をしていないデータ(Vocabのtarget_wordとsentenceのみ)を対象に普通名詞の語彙のみでデータを作成する
# 入力および，出力データはtarget_wordの集約はしていない
# path は　"/data/wikilinks/preprocessed_512tokens_non_ne_vocabwikilinks_more_10_split1.jsonl"  を想定している
def create_common_noun_vocab_df(path, args):
    #nltk.download('punkt')
    #nltk.download('averaged_perceptron_tagger') 
    nlp = spacy.load("en_core_web_sm")
    print(f'前処理していない{path}を対象に普通名詞の語彙のみでデータを作成する')
    df = read_json(path)
    #vocab_list = list(tokenizer.get_vocab().keys())

    ## 処理用データ作成
    print('処理用データ作成')
    target_word_list = []
    sentence_list = []

    # Spacy上でも記号かどうかのチェックはしているので，ここ↓は不要かも
    #code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“” ‘ ’〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％ \/   ]')

    for target_word, sentence in zip(tqdm(df['target_word']), df['sentence']):
        #pos_target_word = nltk.pos_tag([target_word])
        pos_target_word = nlp(target_word)[0].pos #現状はサブワード分割は考慮していない
        #if (pos_target_word == NOUN) and (bool(code_regex.fullmatch(target_word)) == False): #if target_wordが普通名詞 && target_wordが記号ではない
        if (pos_target_word == NOUN) : #if target_wordが普通名詞
            target_word_list.append(target_word)
            sentence_list.append(sentence)
            
    ## wikilinks_df作成
    wikilinks_df = pd.DataFrame(
        data = { 'target_word' : target_word_list, 
                 'sentence': sentence_list
                }
        )
    print(wikilinks_df.head())

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    basename_without_ext = basename_without_ext.replace('preprocessed_', '')
    dir_name = os.path.dirname(path)
    wikilinks_df.to_json(dir_name + '/preprocessed_common_noun_'+ basename_without_ext +'.jsonl', orient='records', force_ascii=False, lines=True)

# Splitされた複数dfのうち，sentenceをtarget_wordの集合に集約させる (df内はtarget_word, sentenceのみ) → (target_word, sentence_list, sentence_count)
def create_aggregation_df(path_list, args):
    df_list = multiple_read_jsonl(path_list)

    ## 処理用データ作成
    print('データを集約')
    sentence_dict = defaultdict(list)
    for df in tqdm(df_list):
        for target_word, sentence in zip(tqdm(df['target_word']), df['sentence']):
            sentence_dict[target_word].append(sentence)

    print('sentence_count 作成')
    sentence_count = [len(s) for s in list(sentence_dict.values())]

    ## _df作成
    aggregated_df = pd.DataFrame(
        data = {'target_word' : list(sentence_dict.keys()), 
                'sentence_list': list(sentence_dict.values()),
                'sentence_count': sentence_count
                }
        )
    print(aggregated_df.head())
    print(f"センテンス数： {aggregated_df['sentence_count'].sum()}")
    print(f"target_word数： {len(aggregated_df['target_word'])}")

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path_list[0])))[0]
    basename_without_ext = basename_without_ext.replace('_split1', '')
    dir_name = os.path.dirname(path_list[0])
    aggregated_df.to_json(dir_name + '/aggregated_'+ basename_without_ext +'.jsonl', orient='records', force_ascii=False, lines=True)


def create_aggregation_df_more_k(path, args):
    aggregated_df = read_json(path)
    THRESHOLD = args.sentence_count_lower
    target_word_list = []
    sentence_list = []
    sentence_count_list = []
    ## ついでにtarget_word中の記号も除去している
    code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“” ‘ ’〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％ \/   ]')
    for target_word, sentence,  sentence_count in zip(tqdm(aggregated_df['target_word']), aggregated_df['sentence_list'], aggregated_df['sentence_count']):
        if sentence_count > THRESHOLD and (bool(code_regex.fullmatch(target_word)) == False):
            target_word_list.append(target_word)
            sentence_list.append(sentence)
            sentence_count_list.append(sentence_count)
        else:
            print(f'しきい値{THRESHOLD}以下なので，削除します')

    print(f'len(target_word_list): {len(target_word_list)}')
    print(f'len(sentence_list): {len(sentence_list)}')
    print(f'len(sentence_count_list): {len(sentence_count_list)}')

    ## df作成
    new_aggregated_df  = pd.DataFrame(
        data = {'target_word' : target_word_list, 
                'sentence_list': sentence_list,
                'sentence_count': sentence_count_list
                }
        )

    print(new_aggregated_df.head())
    print(f"センテンス数： {new_aggregated_df ['sentence_count'].sum()}")
    print(f"target_word数： {len(new_aggregated_df ['target_word'])}")

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    basename_without_ext = basename_without_ext.replace('more_10', '')
    dir_name = os.path.dirname(path)
    new_aggregated_df .to_json(dir_name + '/' + basename_without_ext +'_sentence_more_'+ str(THRESHOLD) +'.jsonl', orient='records', force_ascii=False, lines=True)


# 以前の形式のデータを現行のtarget_word, sentenceの形にする
def create_separete_aggregattion_df(path, args):
    aggregated_df = read_json(path)
    if args.split > 0:
        split_len = args.split
    # 一旦 sentence_listたちの2次元listを一次元にする
    target_word_list = []
    sentence_list = []
    sentence_count_list = []
    word_type_list = []

    for target_word, sentence,  sentence_count, word_type in zip(tqdm(aggregated_df['target_word']), aggregated_df['sentence_list'], aggregated_df['sentence_count'], aggregated_df['word_type']):
        target_word_list += [target_word]*sentence_count
        sentence_list.extend(sentence)
        sentence_count_list += [sentence_count]*sentence_count
        word_type_list += [word_type]*sentence_count

    new_df = pd.DataFrame(
            data = {'target_word' : target_word_list, 
                    'sentence': sentence_list,
                    'sentence_count': sentence_count_list,
                    'word_type' : word_type_list
                    }
    )

    if args.split > 0: #splitして保存する場合
        save_split_jsonl(split_len=split_len, input_df=new_df, output_path=path, is_shuffle=False)
    else :
        new_df.to_json(args.output, orient='records', force_ascii=False, lines=True)
    print("Done")



# NE_SENTENCE_LENと同じくらいのセンテンス数のdfを作成する
# 集約済みデータを対象としている
def create_df_ne_sentence_same_length(path, args):
    aggregated_df = read_json(path)
    aggregated_df = df_shuffle(aggregated_df)
    NE_SENTENCE_LEN =  368954
    target_word_list = []
    sentence_list = []
    sentence_count_list = []
    sentence_count_sum = 0
    
    for target_word, sentence,  sentence_count in zip(tqdm(aggregated_df['target_word']), aggregated_df['sentence_list'], aggregated_df['sentence_count']):
        #sentence count が10以上であるものからNE_SENTENCE_LENに達するまでサンプルする
        if sentence_count_sum <= NE_SENTENCE_LEN:
            #if sentence_count > 10 and target_word != "ads":
            if sentence_count > 10:
                target_word_list.append(target_word)
                sentence_list.append(sentence)
                sentence_count_list.append(sentence_count)
                sentence_count_sum += sentence_count
        else:
            print(f'NE_SENTENCE_LEN： {NE_SENTENCE_LEN}を超えたので，データ追加を終わります')
            print(f'sentence_count_sum = {sentence_count_sum}')
            break
        

    ## df作成
    new_aggregated_df  = pd.DataFrame(
        data = {'target_word' : target_word_list, 
                'sentence_list': sentence_list,
                'sentence_count': sentence_count_list
                }
        )


    print(new_aggregated_df .head())
    print(f"センテンス数： {new_aggregated_df ['sentence_count'].sum()}")
    print(f"target_word数： {len(new_aggregated_df ['target_word'])}")

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    dir_name = os.path.dirname(path)
    new_aggregated_df .to_json(dir_name + '/reduced_' + basename_without_ext + '.jsonl', orient='records', force_ascii=False, lines=True)
    print('saved')

# NE_TARGET_WORD_LENと同じくらいのtarget_word種類数のdfを作成する
def create_df_target_word_same_length(path, args):
    aggregated_df = read_json(path)
    aggregated_df = df_shuffle(aggregated_df)
    NE_TARGET_WORD_LEN =  2219
    target_word_list = []
    sentence_list = []
    sentence_count_list = []
    target_word_sum = 0
    
    for i , (target_word, sentence,  sentence_count) in enumerate(zip(tqdm(aggregated_df['target_word']), aggregated_df['sentence_list'], aggregated_df['sentence_count'])):
        #sentence count が10以上であるものからNE_TARGET_WORD_LENに達するまでサンプルする
        if target_word_sum < NE_TARGET_WORD_LEN:
            if sentence_count > 10 and target_word != "ads":
            #if sentence_count > 10:
                target_word_list.append(target_word)
                sentence_list.append(sentence)
                sentence_count_list.append(sentence_count)
                target_word_sum += 1
        else:
            print(f'NE_TARGET_WORD_LEN： {NE_TARGET_WORD_LEN}を超えたので，データ追加を終わります')
            print(f'target_word_sum = {target_word_sum}')
            break
        

    ## df作成
    new_aggregated_df  = pd.DataFrame(
        data = {'target_word' : target_word_list, 
                'sentence_list': sentence_list,
                'sentence_count': sentence_count_list
                }
        )


    print(new_aggregated_df .head())
    print(f"センテンス数： {new_aggregated_df ['sentence_count'].sum()}")
    print(f"target_word数： {len(new_aggregated_df ['target_word'])}")

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name +'/reduced_targetword/'):  # 無ければdirをつくる
        os.makedirs(dir_name +'/reduced_targetword/') 
    new_aggregated_df .to_json(dir_name + '/reduced_targetword/' + basename_without_ext + '.jsonl', orient='records', force_ascii=False, lines=True)
    print('saved')

# 集約していないdfのColumnにword_typeを追加（例：ne, common_noun）
def add_word_type_column(path, args):
    word_type = args.word_type
    df = read_json(path)
    if 'word_type' not in df.columns :
        df['word_type'] = [word_type for i in range(len(df))]
        df.to_json(path, orient='records', force_ascii=False, lines=True)
    print(f"Done : {path}")


# 集約 &splitされたtensor.ptをconcatする
# NE の集約 & Splitされた tensorをConcatする
def create_concat_tensor(path_list, args):
    print("multiple_load_tensor")
    emb_list = multiple_load_tensor(path_list)
    print(f'type(emb_list) :{type(emb_list)}')
    print(f'len(emb_list) :{len(emb_list)}\n')
    print(f'type(emb_list[0]) :{type(emb_list[0])}')
    print(f'len(emb_list[0]) :{len(emb_list[0])}')
    print(f'len(emb_list[1]) :{len(emb_list[1])}')
    print(f'len(emb_list[2]) :{len(emb_list[2])}\n')
    print(f'type(emb_list[0][0]) :{type(emb_list[0][0])}')
    print(f'len(emb_list[0][0]) :{len(emb_list[0][0])}')
    print(f'len(emb_list[0][1]) :{len(emb_list[0][1])}')
    print(f'len(emb_list[1][1]) :{len(emb_list[1][1])}')
    sentence_emb_list = []
    for emb in tqdm(emb_list):
        sentence_emb_list.extend(emb)
        #for sentence_emb in emb:
        #    sentence_emb_list.extend(sentence_emb)

    del emb_list

    print(f'type(sentence_emb_list) :{type(sentence_emb_list)}')
    print(f"len(sentence_emb_list) :{'{:,}'.format(len(sentence_emb_list))}")
    print(f'type(sentence_emb_list[0]) :{type(sentence_emb_list[0])}')

    emb_tensor = torch.stack(sentence_emb_list, dim=0)  # ここでメモリ不足でkillされる
    del sentence_emb_list
    print(f'type(emb_tensor) :{type(emb_tensor)}')
    print(f'emb_tensor.size() :{emb_tensor.size()}')
    
    path_without_ext = str(path_list[0]).replace('.pt', '')
    dir_name = os.path.dirname(path_list[0])
    output_filename = dir_name + '/concat_'+ basename_without_ext +'.pt'
    torch.save(emb_tensor, output_filename )


# splitされたdfをconcatする
def create_concat_df(path_list, args):
    print("multiple_load_df")
    target_word_list = []
    sentence_list = []
    word_type_list = []
    for path in path_list:
        print(f"reading : {path}")
        df = pd.read_json(path, orient='records', lines=True)
        #print(f"len(df['target_word']) : {len(df['target_word'])}")
        target_word_list.extend(df['target_word'])
        sentence_list.extend(df['sentence'])
        word_type_list.extend(df['word_type'])
        del df
    
    print(f'\n\nlen(target_word_list) : {len(target_word_list)}')
    print(f'len(sentence_list) : {len(sentence_list)}')
    print(f'len(word_type_list) : {len(word_type_list)}')

    new_df = pd.DataFrame(
            data = {'target_word' : target_word_list, 
                    'sentence': sentence_list,
                    'word_type' : word_type_list
                    }
    )
    print(f"saving : {args.output}")
    new_df.to_json(args.output, orient='records', force_ascii=False, lines=True)
    print("Done")

# splitされたdfのColumnをconcatする
def create_concat_column(path_list, args):
    print("multiple_load_df")
    target_word_list = []
    sentence_list = []
    word_type_list = []
    for path in path_list:
        print(f"reading : {path}")
        df = pd.read_json(path, orient='records', lines=True)
        target_word_list.extend(df['target_word'])
        sentence_list.extend(df['sentence'])
        word_type_list.extend(df['word_type'])
        del df
    
    print(f'\n\nlen(target_word_list) : {len(target_word_list)}')
    print(f'len(sentence_list) : {len(sentence_list)}')
    print(f'len(word_type_list) : {len(word_type_list)}')


    basename_without_ext = os.path.splitext(os.path.basename(str(path_list[0])))[0]
    basename_without_ext = basename_without_ext.replace('_split1', '')        
    dir_name = os.path.dirname(path_list[0])
    
    print('saving : '+ dir_name + '/target_word_list_'+ basename_without_ext + '.bin')
    with open(dir_name + '/target_word_list_'+ basename_without_ext + '.bin', mode='wb') as f:
        pickle.dump(target_word_list, f)

    print('saving : '+ dir_name + '/sentence_list_'+ basename_without_ext + '.bin')
    with open(dir_name + '/sentence_list_'+ basename_without_ext + '.bin', mode='wb') as f:
        pickle.dump(sentence_list, f)

    print('saving : '+ dir_name + '/word_type_list_'+ basename_without_ext + '.bin')
    with open(dir_name + '/word_type_list_'+ basename_without_ext + '.bin', mode='wb') as f:
        pickle.dump(word_type_list, f)
    
    print("Done")



def create_delete_symbol_df(path, args):
    print(f'{path} のtarget_wordから記号を削除したdfを作成する')
    df = read_json(path)

    ## 処理用データ作成
    print('処理用データ作成')
    target_word_list = []
    sentence_list = []
    word_type_list = []


    code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“” ‘ ’〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％ \/   ]')

    for target_word, sentence, word_type in zip(tqdm(df['target_word']), df['sentence'], df['word_type']):
        if (bool(code_regex.fullmatch(target_word)) == False): #if target_wordが記号ではない
            target_word_list.append(target_word)
            sentence_list.append(sentence)
            word_type_list.append(word_type)

    new_df = pd.DataFrame(
            data = {'target_word' : target_word_list, 
                    'sentence': sentence_list,
                    'word_type' : word_type_list
                    }
    )

    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    dir_name = os.path.dirname(path)
    output_filename = dir_name + '/delete_symbol_'+ basename_without_ext +'.jsonl'
    print(f"saving : {output_filename}")
    new_df.to_json(output_filename, orient='records', force_ascii=False, lines=True)
    print("Done")

def create_mix_data(target_word_path_list, sentence_path_list, word_type_path_list, SEED, MAX_SENTENCE_NUM, output_path):
    print(f'SEED : {SEED}')
    print(f'MAX_SENTENCE_NUM : {MAX_SENTENCE_NUM}')
    target_word_list = []
    sentence_list = []
    word_type_list = []
    for target_word_path in target_word_path_list:
        with open(target_word_path, 'rb') as f:
            target_word_list.extend(pickle.load(f))
    
    for sentence_path in sentence_path_list:
        with open(sentence_path, 'rb') as f:
            sentence_list.extend(pickle.load(f))
    
    for word_type_path in word_type_path_list:
        with open(word_type_path, 'rb') as f:
            word_type_list.extend(pickle.load(f))

    print(f'\nlen(target_word_list) : {len(target_word_list)}')
    print(f'len(sentence_list) : {len(sentence_list)}')
    print(f'len(word_type_list) : {len(word_type_list)}')

    # ランダムサンプルする
    index_list = list(range(len(target_word_list)))
    random.seed(SEED)
    #MAX_SENTENCE_NUM = 10000000 # 10,000,000 センテンス取得する
    #MAX_SENTENCE_NUM = args.max_sentence_num
    random_sampled_index = random.sample(index_list, MAX_SENTENCE_NUM)
    sampled_target_word_list = [target_word_list[i] for i in random_sampled_index]
    sampled_sentence_list = [sentence_list[i] for i in random_sampled_index]
    sampled_word_type_list = [word_type_list[i] for i in random_sampled_index]

    del target_word_list
    del sentence_list
    del word_type_list

    print(f'\nlen(sampled_target_word_list) : {len(sampled_target_word_list)}')
    print(f'len(sampled_sentence_list) : {len(sampled_sentence_list)}')
    print(f'len(sampled_word_type_list) : {len(sampled_word_type_list)}')

    new_df = pd.DataFrame(
            data = {'target_word' : sampled_target_word_list, 
                    'sentence': sampled_sentence_list,
                    'word_type' : sampled_word_type_list
                    }
    )
    print(f"saving : {output_path}")
    new_df.to_json(output_path, orient='records', force_ascii=False, lines=True)
    print("Done")





args = get_args()

preproc_kind2preproc_func = {
    'extract_df_frequency_more_X' : extract_df_frequency_more_X,
    'data_formating_for_df' : data_formating_for_df,
    'delete_sentence_512tokens_over' : delete_sentence_512tokens_over,
    'save_split_jsonl' : save_split_jsonl,
    'rename_df_columuns' : rename_df_columuns,
    'create_ne_df' : create_ne_df,
    'create_non_ne_vocab_df' : create_non_ne_vocab_df,
    'create_target_word_in_sentence_and_512token_less' : create_target_word_in_sentence_and_512token_less,
    'create_common_noun_vocab_df' : create_common_noun_vocab_df,
    'create_aggregation_df' : create_aggregation_df,
    'create_aggregation_df_more_k' : create_aggregation_df_more_k,
    'create_separete_aggregattion_df' : create_separete_aggregattion_df,
    'create_df_ne_sentence_same_length' : create_df_ne_sentence_same_length,
    'create_df_target_word_same_length' : create_df_target_word_same_length,
    'save_split_jsonl' : save_split_jsonl,
    'add_word_type_column' : add_word_type_column,
    'create_concat_tensor' : create_concat_tensor,
    'create_concat_column' : create_concat_column,
    'create_delete_symbol_df' : create_delete_symbol_df,
    'create_mix_data' : create_mix_data,
}


def preprocess(args):
    for preprocessor in args.preprocessors:
        print(preprocessor)
        preproc_func = preproc_kind2preproc_func[preprocessor]
        if preprocessor in ['create_aggregation_df', 'create_concat_tensor', 'create_concat_column'] :
            preproc_func(args.input_list, args)
        elif preprocessor in ['create_mix_data'] :
            preproc_func(args.target_word_path_list, args.sentence_path_list, args.word_type_path_list, args.SEED, args.MAX_SENTENCE_NUM, args.output)
        elif preprocessor in ['save_split_jsonl']: 
            save_split_jsonl(split_len=args.split, input_jsonl_path=args.input)
        else:
            preproc_func(args.input, args)


preprocess(args)


#print(f'sentence_count_lower = {args.sentence_count_lower}')

#extract_df_frequency_more_X(args.input, args.sentence_count_lower)
#data_formating_for_df(args.sentence_count_lower)
#path = '/data/wikilinks/preprocessed_wikilinks_more_'+str(X)+'.jsonl'
#cut_wiki_df = delete_sentence_512tokens_over(path, args.sentence_count_lower)
# df →　jsonl形式で保存する
#cut_wiki_df.to_json('/data/wikilinks/preprocessed_cuttoken_wikilinks_more_'+str(args.sentence_count_lower)+'.jsonl', orient='records', force_ascii=False, lines=True)

#path = '/data/wikilinks/preprocessed_cuttoken_wikilinks_more_'+str(args.sentence_count_lower)+'.jsonl'
#
#save_split_jsonl(path, args.split)

#path = '/data/wikilinks/preprocessed_cuttoken_wikilinks_more_'+str(args.sentence_count_lower)+'.jsonl'
#
#extract_df_human_and_save(path)

#rename_df_columuns(args.sentence_count_lower)
#data_and_vocab_formating_for_df(args.sentence_count_lower)

#path = '/data/wikilinks/preprocessed_wikilinks_more_'+str(args.sentence_count_lower)+'_vocab.jsonl'
#
#save_split_jsonl(path, args.split)

#path = '/data/wikilinks/preprocessed_wikilinks_more_'+str(args.sentence_count_lower)+'_vocab.jsonl'
#
#cut_wiki_df = delete_sentence_512tokens_over(path, args.sentence_count_lower)
## df →　jsonl形式で保存する
#cut_wiki_df.to_json('/data/wikilinks/preprocessed_512token_wikilinks_more_'+str(args.sentence_count_lower)+'_vocab.jsonl', orient='records', force_ascii=False, lines=True)



#save_split_jsonl(args.input, args.split)

#create_non_ne_vocab_df(args.input, tokenizer)

#rename_df_columuns(args.input, args.before_name, args.after_name)

#create_target_word_in_sentence_and_512token_less(args.input, args.batch_size)

#create_common_noun_vocab_df(args.input)