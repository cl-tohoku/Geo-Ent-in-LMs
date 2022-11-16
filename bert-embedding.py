import argparse
import io,sys
import os
import csv
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
from bokeh.palettes import Category20

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='ne',
                    help='word category. e.g. ne, common_noun')
parser.add_argument("--name_type", type=str, default='lastname',
                    help="using (frist name, last name) pair or (last name)")
parser.add_argument("--L_p", type=int, default='2',
                    help="Setting the L_p norm used in the distance function")
args = parser.parse_args()

def print_distances_clusters(df): 
    print("クラスタ内のベクトル同士の距離の平均")
    print("L1 ")
    capital_ijou_cnt = 0
    bank_ijou_cnt = 0
    pairwise_capital_L1 = 310.33938068547525
    pairwise_bank_L1    = 288.0361706233638
    for target_ne, target_token_embeddings_list, sentence_count	 in zip(df['target_ne'], df['target_token_embeddings_list'], df['sentence_count']):
        result = cal_group_pairwise_Lp_dist_mean(target_token_embeddings_list, p=1)
        print(f'{target_ne}, {result}, {sentence_count}')
        if pairwise_capital_L1 < result:
            capital_ijou_cnt += 1
        if pairwise_bank_L1 < result:
            bank_ijou_cnt += 1
    print(f'capital, {pairwise_capital_L1}, 10')
    print(f'bank, {pairwise_bank_L1}, 10')
    print(f"{len(df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
    print(f"{len(df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")
    print("\n\n")

    print("L2 (2乗)")
    capital_ijou_cnt = 0
    bank_ijou_cnt = 0
    pairwise_capital_L2 = 196.15094833155018
    pairwise_bank_L2    = 171.06298105715894
    for target_ne, target_token_embeddings_list, sentence_count	 in zip(df['target_ne'], df['target_token_embeddings_list'], df['sentence_count']):
        result = cal_group_pairwise_Lp_dist_mean(target_token_embeddings_list, p=2)
        print(f'{target_ne}, {result}, {sentence_count}')
        if pairwise_capital_L2 < result:
            capital_ijou_cnt += 1
        if pairwise_bank_L2 < result:
            bank_ijou_cnt += 1
    print(f'capital, {pairwise_capital_L2}, 10')
    print(f'bank, {pairwise_bank_L2}, 10')
    print(f"{len(df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
    print(f"{len(df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")


    print("クラスタ内の平均ベクトルと各ベクトル間の距離の平均")
    print("L1 ")
    capital_ijou_cnt = 0
    bank_ijou_cnt = 0
    avevec_capital_L1 = 211.02278376019768
    avevec_bank_L1    = 196.26752748101723
    for target_ne, target_token_embeddings_list, average_embeddings, sentence_count	 in zip(df['target_ne'], df['target_token_embeddings_list'], df['average_embeddings'], df['sentence_count']):
        result = cal_group_avevec_Lp_dist_mean(target_token_embeddings_list, average_embeddings, p=1)
        print(f'{target_ne}, {result}, {sentence_count}')
        if avevec_capital_L1 < result:
            capital_ijou_cnt += 1
        if avevec_bank_L1 < result:
            bank_ijou_cnt += 1
    print(f'capital, {avevec_capital_L1}, 10')
    print(f'bank, {avevec_bank_L1}, 10')
    print(f"{len(df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
    print(f"{len(df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")
    print("\n\n")

    print("L2 (2乗)")
    capital_ijou_cnt = 0
    bank_ijou_cnt = 0
    avevec_capital_L2 = 91.01377147239918
    avevec_bank_L2    = 79.54951687964392
    for target_ne, target_token_embeddings_list, average_embeddings, sentence_count	 in zip(df['target_ne'], df['target_token_embeddings_list'], df['average_embeddings'], df['sentence_count']):
        result = cal_group_avevec_Lp_dist_mean(target_token_embeddings_list, average_embeddings, p=2)
        print(f'{target_ne}, {result}, {sentence_count}')
        if avevec_capital_L2 < result:
            capital_ijou_cnt += 1
        if avevec_bank_L2 < result:
            bank_ijou_cnt += 1
    print(f'capital, {avevec_capital_L2}, 10')
    print(f'bank, {avevec_bank_L2}, 10')
    print(f"{len(df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
    print(f"{len(df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")

    ## plot multivariate normal distribution
    #print('多変量正規分布')
    #print('分散共分散行列の対角和')
    #
    #capital_ijou_cnt = 0
    #bank_ijou_cnt = 0
    #covariances_trace_capital = 91.38672402011633
    #covariances_trace_bank    = 80.12135733510718
    #for target_ne, target_token_embeddings_list, sentence_count	 in zip(df['target_ne'], df['target_token_embeddings_list'], df['sentence_count']):
    #    result = cal_mnd_covariances_trace(target_token_embeddings_list)
    #    print(f'{target_ne} context(×{sentence_count}): {result}')
    #    if covariances_trace_capital < result:
    #        capital_ijou_cnt += 1
    #    if covariances_trace_bank < result:
    #        bank_ijou_cnt += 1
    #print(f"{len(df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
    #print(f"{len(df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")
    #print("\n\n")


def cal_mnd_covariances_trace(emb, outfile="GMM.png"):
    # concatenate the two datasets into the final training set
    X_train = emb

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=1, covariance_type="full")
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    ##x = np.linspace(-20.0, 30.0)
    ##y = np.linspace(-20.0, 40.0)
    ##X, Y = np.meshgrid(x, y)
    ##XX = np.array([X.ravel(), Y.ravel()]).T
    ##Z = -clf.score_samples(XX)
    ##Z = Z.reshape(X.shape)
    ##CS = plt.contour(
    ##    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    ##)
    ##CB = plt.colorbar(CS, shrink=0.8, extend="both")

    #plt.figure()
    #plt.scatter(X_train[:, 0], X_train[:, 1])
    #plt.title("Negative log-likelihood predicted by a GMM")
    #plt.axis("tight")
    #plt.savefig(outfile)
    #plt.show()

    #print(f'共分散行列の対角和：{np.trace(clf.covariances_[0])}')
    #print(f'共分散行列の行列式：{LA.det(clf.covariances_[0])}')
    #print(LA.eig(clf.covariances_[0])[0])
    #print(np.sum(LA.eig(clf.covariances_[0])[0]))
    return np.trace(clf.covariances_[0])

def cal_group_pairwise_Lp_dist_mean(embeddings, p=2):
    """
    embeddings: ndarray.  size of ([n,768]) 
    output : int
    """
    L2_dist_list = []
    pdist = torch.nn.PairwiseDistance(p=p)
    embeddings = torch.tensor(embeddings)
    index_combi = list(itertools.combinations(range(embeddings.shape[0]),2))
    for index in index_combi:
        v1 = torch.unsqueeze(embeddings[index[0]], 0)
        v2 = torch.unsqueeze(embeddings[index[1]], 0)
        L2_dist_list.append(pdist(v1, v2))
    L2_dist_mean = torch.cat(L2_dist_list).mean()
    if p==2:
        L2_dist_mean = torch.pow(L2_dist_mean, 2)
    return L2_dist_mean

def cal_group_avevec_Lp_dist_mean(embeddings, ave_vector, p=2):
    """
    embeddings: ndarray.  size of ([n,768]) 
    ave_vector: tensor. size of ([1, 768])
    output : int
    """
    L2_dist_list = []
    pdist = torch.nn.PairwiseDistance(p=p)
    embeddings = torch.tensor(embeddings)
    ave_vector = torch.unsqueeze(ave_vector, 0)
    for emb in embeddings:
        v1 = torch.unsqueeze(emb, 0)
        L2_dist_list.append(pdist(v1, ave_vector))
    L2_dist_mean = torch.cat(L2_dist_list).mean()
    if p==2:
        L2_dist_mean = torch.pow(L2_dist_mean, 2)
    return L2_dist_mean


## 各 Embeddingの最近傍のクラスタの中心が自クラスタである割合を算出
def cal_percentage_of_own_cluster(df, p=2):
    """
    Args:
        tokens_tensor (sbj): Torch tensor size [n_tokens]
            with token ids for each token in text
        or df？
    
    Returns:
        own cluster count: List of int        
        own cluster percentage: List of floats 
    """

    # distance function
    pdist = torch.nn.PairwiseDistance(p=p)
    own_count = 0
    own_count_list = []
    dist_list = []
    percentage_of_own_cluster = []
    for i, (target_token_embeddings_list, sentence_count) in enumerate(zip(df['target_token_embeddings_list'], df['sentence_count'])):
        for target_token_embedding in target_token_embeddings_list:
            target_token_emb = torch.unsqueeze(target_token_embedding, 0)
            for j, ave_embedding in enumerate(df['average_embeddings']):
                ave_emb = torch.unsqueeze(ave_embedding, 0)
                dist_list.extend(pdist(target_token_emb, ave_emb))
            if dist_list.index(min(dist_list)) == i:
                own_count += 1
            dist_list = []

        if args.category == 'ne':
            print(df['target_ne'][i])
        elif args.category == 'common_noun':
            print(df['noun'][i])
        print(f'own_count = {own_count}')
        print(f'sentence_count = {sentence_count}')
        print(f'own_count/sentence_count = {own_count/sentence_count}\n')
        own_count_list.append(own_count)
        percentage_of_own_cluster.append(own_count/sentence_count)
        own_count = 0
    return own_count_list, percentage_of_own_cluster
 


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
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = np.array([token_embed.tolist() for token_embed in token_embeddings])

    return list_token_embeddings


def get_target_token_category(args):
    if args.category == 'ne':
        if args.name_type == 'lastname':
            return 'lastname'
        elif args.name_type == 'firstname_lastname':
            return 'target_ne'
        else :
            raise ValueError("Please enter first and last name pairs or last names for analysis\n e.g. \'--name_type lastname\' or  \'--name_type firstname_lastname\'")
    elif args.category == 'common_noun':
        return 'noun'
    else :
        raise ValueError("Please enter the word category you want to analyze.\n e.g. \'--category ne\' or \'--category common_noun\'")


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

def save_df_to_csv(df, args):
    if os.path.isdir('result/'):
        savefile = 'result/'+str(args.category) +"_L"+str(args.L_p) +"_percentage_of_own_cluster.csv"
    else:
        savefile = str(args.category) +"_L"+str(args.L_p) +"_percentage_of_own_cluster.csv"

    output_df = pd.DataFrame([df[target_token_category], df['percentage_of_own_cluster'], df['own_count_list'],  df['sentence_count']]).T
    micro_ave = cal_micro_ave(df['own_count_list'], df['sentence_count'])
    print(f'micro_ave : {micro_ave}')
    output_df.append({"micro_ave" : micro_ave}, ignore_index=True)
    output_df.to_csv(savefile, encoding="utf_8_sig")


# Loading the pre-trained BERT model
###################################
# Embeddings will be derived from
# the outputs of this model
model = BertModel.from_pretrained('bert-base-uncased',
           output_hidden_states = True,)
# Setting up the tokenizer
###################################
# This is the same tokenizer that
# was used in the model to generate
# embeddings to ensure consistency
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

## dataset install
if args.category == 'ne':
    if args.name_type == 'lastname':
        jsonl_file_path = '/data/ne/human_lastname.jsonl'
    elif args.name_type == 'firstname_lastname':
        jsonl_file_path = '/data/ne/human_firstname_lastname.jsonl'
    else :
        raise ValueError("Please enter first and last name pairs or last names for analysis\n e.g. \'--name_type lastname\' or  \'--name_type firstname_lastname\'")
elif args.category == 'common_noun':
    jsonl_file_path = '/data/common_noun/nouns_sentence.jsonl'
else :
    raise ValueError("Please enter the word category you want to analyze.\n e.g. \'--category ne\' or \'--category common_noun\'")

# TODO : Embedding保存済みのdfをjsonlとして保存→以降それがあれば，それをdfとしてインストール&Embedding計算を省く
processed_jsonl_file_path = '/data/ne/cashed_human_lastname.jsonl'
if os.path.isfile(processed_jsonl_file_path):
    df = pd.read_json(processed_jsonl_file_path, orient='records', lines=True)
else:
    df = pd.read_json(jsonl_file_path, orient="records", lines=True)
    print("Computing Embedding")
    # Getting embeddings for the target
    # word in all given contexts
    target_token_category = get_target_token_category(args)

    target_token_embeddings_list = []
    for target_token , sentence_list, sentence_count in zip(df[target_token_category], df['sentence_list'], df['sentence_count']):
        target_token_sentence_embeddings = []
        tokenized_target_token = tokenizer.tokenize(target_token)

        for sentence in sentence_list:
            tokenized_sentence, sentence_tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
            sentence_embeddings = get_bert_embeddings(sentence_tokens_tensor, segments_tensors, model)
            # Find the position target_token in list of tokens
            #print(tokenized_sentence)
            tokenized_target_token_indexes = [tokenized_sentence.index(t) for t in tokenized_target_token]
            # Get the embedding for target_token
            tokenized_target_token_embeddings = [sentence_embeddings[token_index] for token_index in tokenized_target_token_indexes]
            # サブワード分割されたtokensのAverageベクトルを埋め込みとして使用する            
            ave_target_token_embeddings = generate_average_vector(tokenized_target_token_embeddings)
            target_token_sentence_embeddings.append(ave_target_token_embeddings)
            
        target_token_embeddings_list.append(torch.stack(target_token_sentence_embeddings, dim=0))
        print(f"{target_token} : {tokenized_target_token}, token length : {len(tokenized_target_token_embeddings)}, number of sentences : {sentence_count}")

    df['target_token_embeddings_list'] = target_token_embeddings_list 
    print(len(df))
    print(df['sentence_count'].sum())
    print()

    # generate cluster average vector
    average_embeddings_list = []
    # ここのtarget_token_embeddings_listの名前どうにかしたい
    for embeddings_list in df['target_token_embeddings_list']:
        average_embeddings_list.append(generate_average_vector(embeddings_list))
    df['average_embeddings'] = average_embeddings_list

    #df.to_json('/data/emb_processed_human_more10_lastname_instances_sentences.jsonl', orient='records', force_ascii=False, lines=True)
    ## Error：OverflowError: Maximum recursion level reached



own_count_list , percentage_of_own_cluster = cal_percentage_of_own_cluster(df, args.L_p)
df['own_count_list'] = own_count_list
df['percentage_of_own_cluster'] = percentage_of_own_cluster

save_df_to_csv(df, args)


