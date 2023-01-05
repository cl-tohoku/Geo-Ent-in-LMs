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
#import nltk
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
        if word_type == 'ne':
            category = self.df['notable_figer_types'][idx][0]
            wiki_id = self.df['wiki_id'][idx]
            return target_word, sentence, word_type, category, wiki_id
        else :
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
    parser.add_argument("--emb_path", type=os.path.abspath,  default=None,
                        help="")
    parser.add_argument("--emb_path_list",  nargs='*', default=None,
                            help="")
    parser.add_argument("--output", type=os.path.abspath, 
                        help="save dataset path")
    parser.add_argument("--output_emb", type=os.path.abspath, default=None,
                        help="save embedding path")
    parser.add_argument('--sentence_count_lower', type=int, default=None,
                        help='Lower limit on the number of sentences')
    parser.add_argument('--sentence_count_upper', type=int, default=None,
                  help='Upper limit on the number of sentences')
    parser.add_argument('--alias_count_lower', type=int, default=None,
                        help='Lower limit on the number of sentences')
    parser.add_argument('--alias_count_upper', type=int, default=None,
                  help='Upper limit on the number of sentences')           
    parser.add_argument("--split", type=int, default=0,
                        help="data split")
    parser.add_argument("--before_name", type=str, default=None,
                        help="")
    parser.add_argument("--after_name", type=str, default=None,
                        help="")
    parser.add_argument("--preprocessors", type=str, nargs='+',
                        choices=['extract_df_frequency_more_X',
                                 'create_dataset_formating_for_df',
                                 'delete_sentence_512tokens_over' ,
                                 'save_split_jsonl',
                                 'rename_df_columuns',
                                 'create_target_word_in_sentence_and_512token_less',
                                 'create_common_noun_vocab_df',
                                 'extract_ne_df',
                                 'create_aggregated_df',
                                 'create_aggregated_df_more_k',
                                 'create_separate_df',
                                 'create_alias_ne_df',
                                 'create_df_ne_sentence_same_length',
                                 'create_df_target_word_same_length',
                                 'save_split_jsonl',
                                 'add_word_type_column',
                                 'extractUniqueSentences',
                                 #'save_df2jsonl',
                                 'create_concat_tensor',
                                 'create_concat_column',
                                 'create_delete_symbol_df',
                                 'create_mix_data',
                                 'create_replace_targetWord2title_inSentence',
                                 'create_mixedData2nonNeData',
                                 'samplingData',
                                 'create_various_context_surface_df'],
                        help="")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="")
    parser.add_argument("--is_vocab",  action='store_true',
                        help="")
    parser.add_argument("--is_ne",  action='store_true',
                        help="")
    parser.add_argument("--is_add_title",  action='store_true',
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
    parser.add_argument("--is_group_by_Wiki_id", action='store_true',
                            help="")
    parser.add_argument("--is_separated_df", action='store_true',
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
        #print("error")
        #print(url)
        #print("return None")
        return None
    title = title.replace('_', ' ')
    return title

def remove_parenthesis(text):
    pattern = re.compile(r'\(.*\)')
    result = pattern.sub('', text)
    return result

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
def create_dataset_formating_for_df(path, args):
    wiki_df = read_json(path)
    #print('mention set list 作成')
    #mention_set_list = list(set(wiki_df['mention']))

    ## 処理用データ作成
    print('処理用データ作成')
    target_word_dict = defaultdict(list)
    sentence_dict = defaultdict(list)
    #tokenized_sentence_dict = defaultdict(list)
    notable_figer_types_dict = {}
    #wiki_id_dict = {}

    ## NE として，figer_typesがperson,   location のものを抽出する
    ne_types = ['/person/',   '/location/']

    
    for left_ctx, mention, right_ctx, notable_figer_types, wiki_id in zip(tqdm(wiki_df['left_ctx']), wiki_df['mention'], wiki_df['right_ctx'] , wiki_df['notable_figer_types'] , wiki_df['wiki_id'] ):
        if True in [(ne_type in notable_figer_types[0]) for ne_type in ne_types]: # カテゴリの制限
            sentence = left_ctx + " " +mention + " " +right_ctx
            sentence_dict[wiki_id].append(sentence)
            target_word_dict[wiki_id].append(mention)
            notable_figer_types_dict[wiki_id] = notable_figer_types
            #tokenized_sentence, sentence_tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
            #tokenized_sentence_dict[mention].append(tokenized_sentence)
            #sentence_tokens_tensor_dict[mention].append(sentence_tokens_tensor)
            #segments_tensors_dict[mention].append(segments_tensors)
            #wiki_id_dict[wiki_id] = wiki_id

    print('sentence_count 作成')
    sentence_count = [len(s) for s in list(sentence_dict.values())]

    ## wikilinks_df作成
    wikilinks_df = pd.DataFrame(
        data = {'wiki_id': list(target_word_dict.keys()), 
                'target_word' : list(target_word_dict.values()), 
                'sentence_list': list(sentence_dict.values()),
                'sentence_count': sentence_count,
                'notable_figer_types': list(notable_figer_types_dict.values())
                }
        )
    print(wikilinks_df.head())
    print(f"センテンス数： {wikilinks_df['sentence_count'].sum()}")
    print(f"NE種類数： {len(wikilinks_df['wiki_id'])}")

    # df →　jsonl形式で保存する
    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    dir_name = os.path.dirname(path)
    wikilinks_df.to_json(dir_name + '/dataset2aggregated_df_'+ basename_without_ext +'.jsonl', orient='records', force_ascii=False, lines=True)


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



## 集約済みdfのSentence_listから重複なしセンテンスのみ抽出してdfごと保存する
def extractUniqueSentences(input_jsonl_path, input_emb_path, output_jsonl_path, output_emb_path, args):

    df = read_json(input_jsonl_path)

    emb_list = []
    print(f"input_emb_path : {input_emb_path}")
    if input_emb_path is not None:
        emb_list.extend(torch.load(input_emb_path))
        df['target_word_embeddings_list'] = emb_list


    print(f"is_separated_df : {args.is_separated_df}")
    if args.is_separated_df:
        print(f"is_ne : {args.is_ne}")
        df = aggregate_df(df, is_ne=args.is_ne)


    ## 重複なしセンテンスとそれに対応するtarget_wordを抽出する
    unique_target_word_list = []
    unique_sentence_list = []
    target_word_embeddings_list = []

    for i, (target_word, sentence_list) in enumerate(zip(tqdm(df['target_word']), df['sentence_list'])):
        sentence_set_list = list(set(sentence_list))
        indices = [sentence_list.index(s) for s in sentence_set_list ] 
        if type(target_word) is list :
            unique_target_word_list.append([target_word[index] for index in indices])
        else:
            unique_target_word_list.append(target_word)
        unique_sentence_list.append([sentence_list[index] for index in indices])
        if 'target_word_embeddings_list' in df.columns: 
            target_word_embeddings =  df['target_word_embeddings_list'][i]
            target_word_embeddings_list.extend([target_word_embeddings[index] for index in indices])
    


    # Target_word, Sentence_listを更新
    df = df.assign( sentence_list=unique_sentence_list, \
                    target_word=unique_target_word_list, )

    ## sentence_countの作成
    print('sentence_count 作成')
    new_sentence_count = [len(s) for s in df['sentence_list']]
    df = df.assign(sentence_count=new_sentence_count)
    
    print(f"len(df['sentence_list']) : {len(df['sentence_list'])}")
    print(f"len(df['target_word']) : {len(df['target_word'])}")
    print(f"センテンス数： {'{:,}'.format(df['sentence_count'].sum())}")

    if 'target_word_embeddings_list' in df.columns:
        df = df.drop(columns='target_word_embeddings_list')

    ## dfを保存する
    print(f"saving : {output_jsonl_path}")
    df.to_json(output_jsonl_path, orient='records', force_ascii=False, lines=True)
    print("Done")

    print(f"センテンス数： {df['sentence_count'].sum()}")
    print(f"target_word数： {len(df['target_word'])}\n")
    
    if output_emb_path is not None:
        print(f"len(target_word_embeddings_list)： {len(target_word_embeddings_list)}\n")
        target_word_embeddings_tensor = torch.stack(target_word_embeddings_list)
        torch.save(target_word_embeddings_tensor, output_emb_path)



## (target_word in sentence) のデータを抽出して保存する && バッチ単位で処理 (sentenceが512トークン以下) 
## separeteされたデータをinputしている
def create_target_word_in_sentence_and_512token_less(path, args):
    print("create_target_word_in_sentence_and_512token_less")
    print(f"args.is_ne : {args.is_ne}")
    
    batch_size = args.batch_size
    print(f"batch_size: {batch_size}")
    print("data loading...")
    sentence_dataset = MyDataset(path)
    dataloader = DataLoader(sentence_dataset, batch_size=batch_size)

    # 最大入力長
    INPUT_MAX_LENGTH = 512
    target_word_embeddings_list = []
    target_word_list = []
    sentence_list = []
    word_type_list = []
    category_list = []
    wiki_id_list = []
    tokenized_target_word_len_list = []

    for i, data in enumerate(tqdm(dataloader)):
        if args.is_ne:
            target_word, sentence, word_type, category, wiki_id = data
        else :
            target_word, sentence, word_type = data
        tokenized_target_word = tokenizer(target_word, add_special_tokens=False, return_length=True)
        tokenized_sentence = tokenizer(sentence, max_length=INPUT_MAX_LENGTH, padding=True, truncation=True, return_length=True)
        #tokenized_sentence = tokenizer(sentence, padding=True,  return_length=True)
        for j, (tokenized_sentence_len, tokenized_sentence_ids, tokenized_target_word_ids) in enumerate( zip(tokenized_sentence['length'], tokenized_sentence['input_ids'], tokenized_target_word['input_ids'])):
            ## 512token以内 and sentence中にtarget_wordが含まれるデータのみ抽出 (かつ，現状はサブワードの1tokenのみとしている)
            ## (tokenized_sentence_len <= INPUT_MAX_LENGTH)は別にいらない (max_length=INPUT_MAX_LENGTHとしているので)
            #if (tokenized_sentence_len <= INPUT_MAX_LENGTH) and (tokenized_target_word_ids[0] in tokenized_sentence_ids) and (len(tokenized_target_word_ids) == 1):
            #以下は複数トークンに対応
            if (tokenized_sentence_len <= INPUT_MAX_LENGTH) and all(map(tokenized_sentence_ids.__contains__, tokenized_target_word_ids)):
                ## TODO: ここのindexだけ取得しておいて，indexだけ指定して元のDFからデータを抽出する
                ## これは class MyDatasetのdef __getitem__ をdfのindex返すようにすればいけそう
                
                target_word_list.append(target_word[j])
                sentence_list.append(sentence[j])
                word_type_list.append(word_type[j])
                if args.is_ne:
                    category_list.append(category[j])
                    wiki_id_list.append(wiki_id[j])
                    tokenized_target_word_len_list.append(tokenized_target_word['length'][j])
            else:
                print(f"{j}: センテンスが{INPUT_MAX_LENGTH}トークン以上 or センテンス中にtarget_wordが含まれていません ")
                print('skipします')


    new_df = pd.DataFrame(
        data = {'target_word' : target_word_list, 
                'sentence': sentence_list,
                'word_type' : word_type_list
                }
        )
    if args.is_ne:
        new_df['notable_figer_types'] = category_list
        new_df['wiki_id'] = wiki_id_list
        new_df['target_word_sub_len'] = tokenized_target_word_len_list

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


#TODO:  前処理済みdfからBERTs の語彙との共通部分  OR  人名であるものを抽出して保存する
#TODO: create_dataset_formating_for_dfと統合する → 処理が遅いので，extract_df_human_and_saveっぽくする．後にextract_df_human_and_saveを統合する → argsでコントロールする
# target_wordの集約されている (以前の形式)
def extract_ne_df(path, args, tokenizer=tokenizer):
    ## CutTokenじゃないので注意
    #path = '/data/wikilinks/preprocessed_wikilinks_more_'+str(args.sentence_count_lower)+'.jsonl'
    if args.is_vocab == False and args.is_add_title == False and args.sentence_count_lower is None:
        raise ValueError("is_vocab and is_add_title are False")
    
    df = read_json(path)
    
    
    if args.is_vocab:
        vocab_list = list(tokenizer.get_vocab().keys())
    
    print(df.head())

    ## 処理用データ作成
    print('処理用データ作成')
    title_list = []
    extract_target_word = []
    extract_sentence_list = []
    extract_wiki_id = []
    extract_notable_figer_types = []

    ## NE として，figer_typesがperson,   location のものを抽出する
    ne_types = ['/person/',   '/location/']

    for target_word, sentence_list, sentence_count, wiki_id, notable_figer_types in zip( tqdm(df['target_word']), df['sentence_list'], df['sentence_count'], df['wiki_id'], df['notable_figer_types']):
        ## ne_typesのものだけを抽出する
        if (args.is_vocab == True) and (target_word in vocab_list) and (True in [(ne_type in notable_figer_types[0]) for ne_type in ne_types]):
            extract_target_word.append(target_word)
            extract_sentence_list.append(sentence_list)
            extract_wiki_id.append(wiki_id)
            extract_notable_figer_types.append(notable_figer_types)
        ## title columnを追加する
        elif args.is_add_title == True:
            title = getTitleFromWikipediaURL(wiki_id)
            if title is not None:
                title = remove_parenthesis(title)
                if title != '':
                    title_list.append(title)
                    extract_target_word.append(target_word)
                    extract_sentence_list.append(sentence_list)
                    extract_wiki_id.append(wiki_id)
                    extract_notable_figer_types.append(notable_figer_types)
        ## センテンス数が THRESHOLD より多いデータのみ追加する
        elif args.sentence_count_lower is not None:
            THRESHOLD = args.sentence_count_lower
            if sentence_count > THRESHOLD:
                extract_target_word.append(target_word)
                extract_sentence_list.append(sentence_list)
                extract_wiki_id.append(wiki_id)
                extract_notable_figer_types.append(notable_figer_types)
            else:
                print(f'しきい値{THRESHOLD}以下なので，追加しません')


    print('sentence_count 作成')
    extract_sentence_count = [len(s) for s in extract_sentence_list]

    ## wikilinks_df作成
    wikilinks_df = pd.DataFrame(
        data = {'target_word' : extract_target_word, 
                'sentence_list': extract_sentence_list,
                'sentence_count': extract_sentence_count,
                'wiki_id': extract_wiki_id, 
                'notable_figer_types': extract_notable_figer_types
                }
        )
    
    if args.is_add_title:
        wikilinks_df['title'] = title_list
    
    print(f"センテンス数： {'{:,}'.format(wikilinks_df['sentence_count'].sum())}")
    print(f"NE種類数： {'{:,}'.format(len(wikilinks_df['wiki_id']))}")
    print(wikilinks_df.head())
    basename = path.replace('.jsonl', '')
    # df →　jsonl形式で保存する
    if args.is_vocab:
        wikilinks_df.to_json(basename +'_NE_vocab.jsonl', orient='records', force_ascii=False, lines=True)
    elif args.is_add_title:
        wikilinks_df.to_json(basename +'_title.jsonl', orient='records', force_ascii=False, lines=True)
    elif args.sentence_count_lower is not None:
        wikilinks_df.to_json(basename +'_more'+str(args.sentence_count_lower)+'.jsonl', orient='records', force_ascii=False, lines=True)


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

# Splitされた複数dfのうち，sentenceをtarget_wordの集合に集約させる (df内はtarget_word, sentence, word_typeの3列) → (target_word, sentence_list, sentence_count, word_type)
# 後にConcat_dfと合わせる
#def create_aggregated_df(path_list, args):
#    df_list = multiple_read_jsonl(path_list)
#
#    ## 処理用データ作成
#    print('データを集約')
#    sentence_dict = defaultdict(list)
#    word_type_dict = {}
#    for df in tqdm(df_list):
#        for target_word, sentence, word_type in zip(tqdm(df['target_word']), df['sentence'], df['word_type']):
#            sentence_dict[target_word].append(sentence)
#            word_type_dict[target_word] = word_type
#
#    print('sentence_count 作成')
#    sentence_count = [len(s) for s in list(sentence_dict.values())]
#
#    ## _df作成
#    aggregated_df = pd.DataFrame(
#        data = {'target_word' : list(sentence_dict.keys()), 
#                'sentence_list': list(sentence_dict.values()),
#                'sentence_count': sentence_count,
#                'word_type' : list(word_type_dict.values())
#                }
#        )
#    print(aggregated_df.head())
#    print(f"センテンス数： {aggregated_df['sentence_count'].sum()}")
#    print(f"target_word種類数： {len(aggregated_df['target_word'])}")
#
#    # df →　jsonl形式で保存する
#    if args.output is not None:
#        file_name = args.output
#    else :
#        basename_without_ext = os.path.splitext(os.path.basename(str(path_list[0])))[0]
#        basename_without_ext = basename_without_ext.replace('_split1', '')
#        dir_name = os.path.dirname(path_list[0])
#        file_name = dir_name + '/aggregated_'+ basename_without_ext +'.jsonl'
#
#    aggregated_df.to_json(file_name, orient='records', force_ascii=False, lines=True)


def create_aggregated_df_more_k(path, args):
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
def create_separate_df(path, args):
    aggregated_df = read_json(path)
    if args.split > 0:
        split_len = args.split
    # 一旦 sentence_listたちの2次元listを一次元にする
    target_word_list = []
    sentence_list = []
    sentence_count_list = []
    word_type_list = []
    category_list = []
    wiki_id_list = []
    target_word_sub_len_list = []
    alias_count_list = []

    for i, (target_word, sentence,  sentence_count, word_type) in enumerate(zip(tqdm(aggregated_df['target_word']), aggregated_df['sentence_list'], aggregated_df['sentence_count'], aggregated_df['word_type'])):
        if 'alias_count'  in aggregated_df.columns:
            target_word_list += target_word
        else:
            target_word_list += [target_word]*sentence_count
        sentence_list.extend(sentence)
        sentence_count_list += [sentence_count]*sentence_count
        word_type_list += [word_type]*sentence_count
        if args.is_ne:
            category_list += [aggregated_df['notable_figer_types'][i]]*sentence_count
            wiki_id_list += [aggregated_df['wiki_id'][i]]*sentence_count
            if 'target_word_sub_len'  in aggregated_df.columns:
                target_word_sub_len_list += [aggregated_df['target_word_sub_len'][i]]*sentence_count
            if 'alias_count'  in aggregated_df.columns:
                alias_count_list += [aggregated_df['alias_count'][i]]*sentence_count
                
    print(f"len(target_word_list) : {len(target_word_list)}")
    print(f"len(sentence_list) : {len(sentence_list)}")
    print(f"len(sentence_count) : {len(sentence_count_list)}")
    print(f"len(word_type) : {len(word_type_list)}")

    new_df = pd.DataFrame(
            data = {'target_word' : target_word_list, 
                    'sentence': sentence_list,
                    'sentence_count': sentence_count_list,
                    'word_type' : word_type_list
                    }
    )

    if args.is_ne:
        new_df['notable_figer_types'] = category_list
        new_df['wiki_id'] = wiki_id_list
        if 'target_word_sub_len'  in aggregated_df.columns:
            new_df['target_word_sub_len'] = target_word_sub_len_list
        if 'alias_count'  in aggregated_df.columns:
            new_df['alias_count'] = alias_count_list 


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
    new_aggregated_df.to_json(dir_name + '/reduced_' + basename_without_ext + '.jsonl', orient='records', force_ascii=False, lines=True)
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

# dfのColumnにword_typeを追加（例：ne, common_noun, non_ne）
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
    #print(f'type(emb_list) :{type(emb_list)}')
    #print(f'len(emb_list) :{len(emb_list)}\n')
    #print(f'type(emb_list[0]) :{type(emb_list[0])}')
    #print(f'len(emb_list[0]) :{len(emb_list[0])}')
    #print(f'len(emb_list[1]) :{len(emb_list[1])}')
    #print(f'len(emb_list[2]) :{len(emb_list[2])}\n')
    #print(f'type(emb_list[0][0]) :{type(emb_list[0][0])}')
    #print(f'len(emb_list[0][0]) :{len(emb_list[0][0])}')
    #print(f'len(emb_list[0][1]) :{len(emb_list[0][1])}')
    #print(f'len(emb_list[1][1]) :{len(emb_list[1][1])}')
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

# splitされたdfのColumnをconcatしてpickleで保存する
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


def create_replace_targetWord2title_inSentence(path, args):
    print(f'{path} のtarget_wordとSentence中の単語をtitleに置換する')
    df = read_json(path)

    ## 処理用データ作成
    print('処理用データ作成')
    replaced_sentence_list = []
    replaced_target_word_list = []

    for target_word_list, sentence_list, title in zip(tqdm(df['target_word']), df['sentence_list'], df['title']):
        replaced_sentences = []
        for target_word, sentence in zip(target_word_list, sentence_list):
            replaced_sentence = sentence.replace(target_word, title)
            replaced_sentences.append(replaced_sentence)
        replaced_sentence_list.append(replaced_sentences)
        replaced_target_word_list.append(title)

    df['target_word'] = replaced_target_word_list
    df['sentence_list'] = replaced_sentence_list


    basename_without_ext = os.path.splitext(os.path.basename(str(path)))[0]
    dir_name = os.path.dirname(path)
    output_filename = dir_name + '/replaced_'+ basename_without_ext +'.jsonl'
    print(f"saving : {output_filename}")
    df.to_json(output_filename, orient='records', force_ascii=False, lines=True)
    print("Done")


def create_mixedData2nonNeData(input_list, emb_path_list, output_path, args):
    target_word_list = []
    sentence_list = []
    word_type_list = []

    ## dataset install
    df_list = multiple_read_jsonl(input_list)
    emb_list = multiple_load_tensor(emb_path_list)

    ## splitされたデータをconcatする
    concat_df = pd.concat(df_list).reset_index(drop=True)
    concat_df['target_word_embeddings_list'] = emb_list


    print(f"concat_df['target_word']: {len(concat_df['target_word'])}")
    print(f"concat_df['target_word_embeddings_list']: {len(concat_df['target_word_embeddings_list'])}")

    non_ne_df = concat_df[concat_df['word_type'] == 'non_ne'].reset_index(drop=True)
   

    new_df = pd.DataFrame(
            data = {'target_word' : non_ne_df['target_word'], 
                    'sentence': non_ne_df['sentence'],
                    'word_type' : non_ne_df['word_type']
                    }
    )

    jsonl_output_path = output_path
    print(f"saving : {jsonl_output_path}")
    new_df.to_json(jsonl_output_path, orient='records', force_ascii=False, lines=True)

    emb_tensor = torch.stack(non_ne_df['target_word_embeddings_list'].tolist(), dim=0)
    tensor_output_path = str(jsonl_output_path).replace('.jsonl', '')
    torch.save(emb_tensor, tensor_output_path +  "_tensor.pt")

    print("Done")

# input: separeted df or aggregate_df , (optional: separeted tensor)
# output:  aggregated df (optional: separeted tensor)
# クラスタ内の点群は必ず10以上となっているようにフィルターをかける (つつ，サンプルしている？)

def samplingData(input_jsonl_path, input_emb_path, output_jsonl_path, output_emb_path, args):
    print(f"is_ne : {args.is_ne}")
    print(f"is_group_by_Wiki_id : {args.is_group_by_Wiki_id}")
    print(f"is_separated_df : {args.is_separated_df}")

    df = read_json(input_jsonl_path)

    emb_list = []
    print(f"input_emb_path : {input_emb_path}")
    if input_emb_path is not None:
        emb_list.extend(torch.load(input_emb_path))
        df['target_word_embeddings_list'] = emb_list
    #concat_df['average_embeddings'] = concat_ave_emb

    
    # dfを集約する
    if args.is_separated_df:
        aggregated_df = aggregate_df(df, is_ne=args.is_ne)
    else :
        aggregated_df = df
    aggregated_df = df_shuffle(aggregated_df)
    SAMPLING_LOWER = 10
    SENTENCE_COUNT_UPPER =  args.sentence_count_upper
    sentence_count_sum = 0
    print(f"SENTENCE_COUNT_UPPER : {SENTENCE_COUNT_UPPER}")

    #sentence_dict = defaultdict(list)
    target_word_dict = defaultdict(list)
    sentence_dict = {}
    target_word_embeddings_dict = defaultdict(list)
    target_word_embeddings_list = []
    word_type_dict = {}
    category_dict = {}
    wiki_id_dict = {}
    target_word_sub_len_dict = {}
    alias_count_dict = {}

    for i, (target_word, sentence_list,  word_type, sentence_count) in enumerate(zip(tqdm(aggregated_df['target_word']), aggregated_df['sentence_list'], aggregated_df['word_type'], aggregated_df['sentence_count'])):
        if sentence_count < SAMPLING_LOWER:
            continue
        if sentence_count_sum >= SENTENCE_COUNT_UPPER:
            print(f'NE_SENTENCE_LEN： {SENTENCE_COUNT_UPPER}を超えたので，データ追加を終わります')
            print(f'sentence_count_sum = {sentence_count_sum}')
            break
        if args.is_ne:
            sample_num = random.randint(SAMPLING_LOWER, sentence_count)
        else:
            sample_num = sentence_count - 1
        sentence_count_sum += sample_num
    
        if args.is_group_by_Wiki_id:
            wiki_id = aggregated_df['wiki_id'][i]
            key = wiki_id
        else :
            key = target_word

        #target_word_dict[key].extend(target_word)
        target_word_dict[key] = target_word[0:sample_num]
        sentence_dict[key] = sentence_list[0:sample_num]
        if 'target_word_embeddings_list' in aggregated_df.columns: 
            target_word_embeddings =  aggregated_df['target_word_embeddings_list'][i]
            #target_word_embeddings_dict[key].append(target_word_embeddings[0:sample_num])
            target_word_embeddings_list.extend(target_word_embeddings[0:sample_num])
        word_type_dict[key] = word_type

        if args.is_ne:
            category_dict[key] = aggregated_df['notable_figer_types'][i]
            wiki_id_dict[key] = aggregated_df['wiki_id'][i]
            if 'target_word_sub_len' in aggregated_df.columns:
                target_word_sub_len_dict[key] = aggregated_df['target_word_sub_len'][i]



    print('新たなsentence_count 作成')
    new_sentence_count = [len(s) for s in list(sentence_dict.values())]


    ## df作成
    sampled_aggregated_df = pd.DataFrame(
        data = {
                'sentence_list': list(sentence_dict.values()),
                'sentence_count': new_sentence_count,
                #'target_word_embeddings_list' : list(target_word_embeddings_dict.values()),
                'word_type' : list(word_type_dict.values())
                }
        )
    if args.is_ne:
        sampled_aggregated_df['notable_figer_types'] = list(category_dict.values())
        sampled_aggregated_df['wiki_id'] = list(wiki_id_dict.values())
        if 'target_word_sub_len' in aggregated_df.columns:
            sampled_aggregated_df['target_word_sub_len'] = list(target_word_sub_len_dict.values())

    if args.is_group_by_Wiki_id: # group by wiki_id
        sampled_aggregated_df['target_word'] = list(target_word_dict.values())
    else: # group by target_word
        sampled_aggregated_df['target_word'] = list(sentence_dict.keys())

    # alias_count作成
    if 'alias_count' in aggregated_df.columns:
        print('新たなalias_count 作成')
        new_alias_count = [len(list(set(w))) for w in sampled_aggregated_df['target_word']]
        sampled_aggregated_df['alias_count'] = new_alias_count

    print(sampled_aggregated_df.head())
    print(f"センテンス数： {sampled_aggregated_df['sentence_count'].sum()}")
    print(f"target_word数： {len(sampled_aggregated_df['target_word'])}\n")
    print(f"sentence_count_sum ： {sentence_count_sum }\n")
    if 'target_word_embeddings_list' in aggregated_df.columns:
        print(f"len(target_word_embeddings_list)： {len(target_word_embeddings_list)}\n")

    # save 
    print(f"saving : {output_jsonl_path}")
    sampled_aggregated_df.to_json(output_jsonl_path, orient='records', force_ascii=False, lines=True)

    ## tensorだけ別で1次元で保存する
    if 'target_word_embeddings_list' in aggregated_df.columns:
        target_word_embeddings_tensor = torch.stack(target_word_embeddings_list)
        torch.save(target_word_embeddings_tensor, output_emb_path)


    #for target_word, sentence,  sentence_count in zip(tqdm(aggregated_df['target_word']), aggregated_df['sentence_list'], aggregated_df['sentence_count']):
    #    if sentence_count_sum >= SENTENCE_COUNT_UPPER:
    #        break
    #    sample_num = random.randint(SAMPLING_LOWER, sentence_count)
    #    sentence_count_sum += sample_num
        
    



def aggregate_df(df, is_ne=False):
    ## 処理用データ作成
    print('データを集約')
    sentence_dict = defaultdict(list)
    target_word_embeddings_dict = defaultdict(list)
    word_type_dict = {}
    category_dict = {}
    wiki_id_dict = {}
    target_word_sub_len_dict = {}

    for i, (target_word, sentence, target_word_embedding, word_type) in enumerate(zip(tqdm(df['target_word']), df['sentence'], df['target_word_embeddings_list'], df['word_type'])):
        sentence_dict[target_word].append(sentence)
        target_word_embeddings_dict[target_word].append(target_word_embedding)
        word_type_dict[target_word] = word_type
        if args.is_ne:
            category_dict[target_word] = df['notable_figer_types'][i]
            wiki_id_dict[target_word] = df['wiki_id'][i]
            target_word_sub_len_dict[target_word] = df['target_word_sub_len'][i]

        #target_word_embeddings_dict[target_word].append(torch.stack(target_word_embedding, dim=0))

    print('sentence_count 作成')
    sentence_count = [len(s) for s in list(sentence_dict.values())]

    ## df作成
    aggregated_df = pd.DataFrame(
        data = {'target_word' : list(sentence_dict.keys()), 
                'sentence_list': list(sentence_dict.values()),
                'sentence_count': sentence_count,
                'target_word_embeddings_list' : list(target_word_embeddings_dict.values()),
                'word_type' : list(word_type_dict.values())
                }
        )
    if args.is_ne:
        aggregated_df['notable_figer_types'] = list(category_dict.values())
        aggregated_df['wiki_id'] = list(wiki_id_dict.values())
        aggregated_df['target_word_sub_len'] = list(target_word_sub_len_dict.values())

    print(aggregated_df.head())
    print(f"センテンス数： {aggregated_df['sentence_count'].sum()}")
    print(f"target_word数： {len(aggregated_df['target_word'])}\n")

    return aggregated_df



def create_alias_ne_df(input_path, output_path, args):
    """
    input : aggregated df (NE)
    """
    ne_df = read_json(input_path)
    ne_df = df_shuffle(ne_df)

    ALIAS_COUNT_LOWER = args.alias_count_lower
    ALIAS_COUNT_UPPER = args.alias_count_upper
    SENTENCE_SAMPLE_SIZE = 3
    SENTENCE_COUNT_UPPER = args.sentence_count_upper


    ## alias数 [alias_count_lower～alias_count_upper]の範囲のNEを抽出
    extracted_ne_df = ne_df[((ne_df['alias_count'] >= ALIAS_COUNT_LOWER) & (ne_df['alias_count'] <= ALIAS_COUNT_UPPER))]

    sentence_count_sum = 0
    ## alias数 * SENTENCE_SAMPLE_SIZE >= SENTENCE_COUNT_UPPER くらいデータをサンプルする (16,056くらい)
    for i, alias_count in enumerate(extracted_ne_df['alias_count']):
        sentence_count_sum += alias_count * SENTENCE_SAMPLE_SIZE
        if  sentence_count_sum  >= SENTENCE_COUNT_UPPER:
            sample_index = i
            break

    print(f"SENTENCE_COUNT_UPPER : {SENTENCE_COUNT_UPPER}")
    print(f"sentence_count_sum : {sentence_count_sum}")
    print(f"sample_index : {sample_index}")
    sampled_ne_df = extracted_ne_df[0:sample_index+1]


    ## sentence_set_list 中の SENTENCE_SAMPLE_SIZE 個のセンテンスをサンプルする (センテンスの重複なし)
    sampled_target_word_sentence_list = []
    for target_word, sentence_list in zip(tqdm(sampled_ne_df['target_word']), sampled_ne_df['sentence_list']):
        sampled_target_word_sentence = []
        sentence_set_list = list(set(sentence_list))
        for k, sentece in enumerate(sentence_set_list): 
            if k >= SENTENCE_SAMPLE_SIZE:
                break
            index = sentence_list.index(sentece)
            sampled_target_word_sentence.append((target_word[index], sentence_list[index]))
        sampled_target_word_sentence_list.append(sampled_target_word_sentence)


    ## target_wordをaliasに置換する 
    
    new_sentences_list = [] 
    new_target_word_list = [] 
    for target_word, sampled_target_word_sentence in zip(tqdm(sampled_ne_df['target_word']), sampled_target_word_sentence_list): 
        new_sentences = [] 
        new_target_word = [] 
        alias_list = list(set(target_word))
        for target_word_sentence_tuple in sampled_target_word_sentence:
            target_word = target_word_sentence_tuple[0]
            sentence = target_word_sentence_tuple[1]
            for alias in alias_list:
                new_sentences.append(sentence.replace(target_word, alias))
                new_target_word.append(alias)
        new_sentences_list.append(new_sentences)
        new_target_word_list.append(new_target_word)

    print(f"len(new_sentences_list) : {len(new_sentences_list)}")
    print(f"len(new_target_word_list) : {len(new_target_word_list)}")
    
    ## sentence_list&target_wordの差し替え 
    sampled_ne_df = sampled_ne_df.assign(sentence_list=new_sentences_list, \
                                         target_word=new_target_word_list, )

    ## sentence_countの作成
    print('sentence_count 作成')
    new_sentence_count = [len(s) for s in sampled_ne_df['sentence_list']]
    sampled_ne_df = sampled_ne_df.assign(sentence_count=new_sentence_count)
    
    print(f"len(sampled_ne_df['sentence_list']) : {len(sampled_ne_df['sentence_list'])}")
    print(f"len(sampled_ne_df['target_word']) : {len(sampled_ne_df['target_word'])}")
    print(f"センテンス数： {'{:,}'.format(sampled_ne_df['sentence_count'].sum())}")

    ## dfを保存する
    print(f"saving : {output_path}")
    sampled_ne_df.to_json(output_path, orient='records', force_ascii=False, lines=True)
    print("Done")


def create_various_context_surface_df(input_path_list, output_path, args):
    """
    input :  jsonl list (origin,  various_context)
    """

    df_list = multiple_read_jsonl(input_path_list)
    origin_df = df_list[0]
    various_context_df = df_list[1]

    extract_sentence_count = various_context_df['sentence_count']
    extract_wiki_id = various_context_df['wiki_id']
    extract_ne_df = origin_df[origin_df['wiki_id'].isin(extract_wiki_id)]

    new_target_word = []
    new_sentences_list = []

    print(len(extract_sentence_count))
    print(len(extract_ne_df['target_word']))
    print(len(extract_ne_df['sentence_list']))
    

    for target_word, sentence_list, wiki_id in zip(extract_ne_df['target_word'], extract_ne_df['sentence_list'], tqdm(extract_ne_df['wiki_id'])):
        #print(f"sentence_count : {sentence_count}, \t len(sentence_list) : {len(sentence_list)}")
        for various_context_wiki_id, various_context_sentence_count in zip(various_context_df['wiki_id'],various_context_df['sentence_count']) :
            if wiki_id == various_context_wiki_id:
                new_target_word.append(target_word[0:various_context_sentence_count])
                new_sentences_list.append(sentence_list[0:various_context_sentence_count])
        #new_target_word.append(target_word)
        #new_sentences_list.append(sentence_list)

    print('sentence_count 作成')
    new_sentence_count = [len(s) for s in new_sentences_list]
    print('新たなalias_count 作成')
    new_alias_count = [len(list(set(w))) for w in new_target_word]
    print(f"センテンス数： {'{:,}'.format(sum(new_sentence_count))}")
    #extract_ne_df.assign(sentence_list=new_sentences_list, \
    #                     target_word=new_target_word, \
    #                     sentence_count=new_sentence_count)
    extract_ne_df['target_word'] = new_target_word
    extract_ne_df['sentence_list'] = new_sentences_list
    extract_ne_df['sentence_count'] = new_sentence_count
    extract_ne_df['alias_count'] = new_alias_count

    print(f"センテンス数： {'{:,}'.format(extract_ne_df['sentence_count'].sum())}")


    ## dfを保存する
    print(f"saving : {output_path}")
    extract_ne_df.to_json(output_path, orient='records', force_ascii=False, lines=True)
    print("Done")


args = get_args()

preproc_kind2preproc_func = {
    'extract_df_frequency_more_X' : extract_df_frequency_more_X,
    'create_dataset_formating_for_df' : create_dataset_formating_for_df,
    'delete_sentence_512tokens_over' : delete_sentence_512tokens_over,
    'save_split_jsonl' : save_split_jsonl,
    'rename_df_columuns' : rename_df_columuns,
    'extract_ne_df' : extract_ne_df,
    'create_non_ne_vocab_df' : create_non_ne_vocab_df,
    'create_target_word_in_sentence_and_512token_less' : create_target_word_in_sentence_and_512token_less,
    'create_common_noun_vocab_df' : create_common_noun_vocab_df,
    'create_alias_ne_df' : create_alias_ne_df,
    #'create_aggregated_df' : create_aggregated_df,
    'create_aggregated_df_more_k' : create_aggregated_df_more_k,
    'create_separate_df' : create_separate_df,
    'create_df_ne_sentence_same_length' : create_df_ne_sentence_same_length,
    'create_df_target_word_same_length' : create_df_target_word_same_length,
    'save_split_jsonl' : save_split_jsonl,
    'add_word_type_column' : add_word_type_column,
    'create_concat_tensor' : create_concat_tensor,
    'create_concat_column' : create_concat_column,
    'create_delete_symbol_df' : create_delete_symbol_df,
    'create_mix_data' : create_mix_data,
    'create_replace_targetWord2title_inSentence' : create_replace_targetWord2title_inSentence,
    'create_mixedData2nonNeData' : create_mixedData2nonNeData,
    'samplingData' : samplingData,
    'extractUniqueSentences' :extractUniqueSentences,
    'create_various_context_surface_df' : create_various_context_surface_df,
}


def preprocess(args):
    for preprocessor in args.preprocessors:
        print(preprocessor)
        preproc_func = preproc_kind2preproc_func[preprocessor]
        if preprocessor in ['create_aggregated_df', 'create_concat_tensor', 'create_concat_column'] :
            preproc_func(args.input_list, args)
        elif preprocessor in ['create_alias_ne_df'] :
            preproc_func(args.input, args.output, args)
        elif preprocessor in ['create_mix_data'] :
            preproc_func(args.target_word_path_list, args.sentence_path_list, args.word_type_path_list, args.SEED, args.MAX_SENTENCE_NUM, args.output)
        elif preprocessor in ['create_mixedData2nonNeData'] :
            preproc_func(args.input_list, args.emb_path_list, args.output, args)
        elif preprocessor in ['create_various_context_surface_df'] :
            preproc_func(args.input_list,  args.output, args)
        elif preprocessor in ['samplingData', 'extractUniqueSentences'] :
            preproc_func(input_jsonl_path=args.input, input_emb_path=args.emb_path, output_jsonl_path=args.output, output_emb_path=args.output_emb, args=args)
        elif preprocessor in ['save_split_jsonl']: 
            save_split_jsonl(split_len=args.split, input_jsonl_path=args.input)
        else:
            preproc_func(args.input, args)


preprocess(args)