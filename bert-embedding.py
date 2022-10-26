import argparse
import os
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


def cal_mnd_covariances_trace(emb, outfile="GMM.png"):
    # concatenate the two datasets into the final training set
    #X_train = np.vstack([shifted_gaussian, stretched_gaussian])
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


#parser = argparse.ArgumentParser()
#parser.add_argument('--test_file', type=str, default=None)
#args = parser.parse_args()

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

## data install
# TODO : Embedding保存済みのdfをjsonlとして保存→以降それがあれば，それをdfとしてインストール&Embedding計算を省く
processed_jsonl_file_path = '/data/emb_processed_human_more10_lastname_instances_sentences.jsonl'
if os.path.isfile(processed_jsonl_file_path):
    human_df = pd.read_json('/data/emb_processed_human_more10_lastname_instances_sentences.jsonl', orient='records', lines=True)
else:
    #human_df = pd.read_json('/data/human_more10_lastname_instances_sentences.jsonl', orient='records', lines=True)
    human_df = pd.read_json("/data/human_more10_lastname_instances_sentences_addlastname_subword_less3.jsonl", orient="records", lines=True)
    print("Computing Embedding")
    # Getting embeddings for the target
    # word in all given contexts

    target_token_embeddings_list = []
    for target_ne, sentence__list, lastname in zip(human_df['target_ne'], human_df['sentence_list'], human_df['lastname']):
        target_token_sentence_embeddings = []
        target_token = lastname
        tokenized_target_token = tokenizer.tokenize(target_token)
        print(tokenized_target_token)
        if len(tokenized_target_token) >= 4:
            print(f'len(tokenized_target_token) >= 4: {tokenized_target_token}')
            continue

        for sentence_dict in sentence__list:
            sentence = sentence_dict['sentence']
            tokenized_sentence, sentence_tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
            sentence_embeddings = get_bert_embeddings(sentence_tokens_tensor, segments_tensors, model)
            # Find the position target_token in list of tokens
            tokenized_target_token_indexes = [tokenized_sentence.index(t) for t in tokenized_target_token]
            # Get the embedding for target_token
            tokenized_target_token_embeddings = [sentence_embeddings[token_index] for token_index in tokenized_target_token_indexes]
            # サブワード分割されたtokensのAverageベクトルを埋め込みとして使用する            
            ave_target_token_embeddings = generate_average_vector(tokenized_target_token_embeddings)
            sentence_dict['embedding'] = ave_target_token_embeddings
            target_token_sentence_embeddings.append(ave_target_token_embeddings)
        target_token_embeddings_list.append(torch.stack(target_token_sentence_embeddings, dim=0))

    human_df['target_token_embeddings_list'] = target_token_embeddings_list 
    print(len(human_df))
    print(human_df['sentence_count'].sum())
    print()

    # generate average vector
    average_embeddings_list = []
    # ここのtarget_token_embeddings_listの名前どうにかする
    for embeddings_list in human_df['target_token_embeddings_list']:
        print('print(np.array(embeddings_list).shape)')
        print(np.array(embeddings_list).shape)
        print(type(embeddings_list))
        print(type(embeddings_list[0]))
        #print(embeddings_list)
        average_embeddings_list.append(generate_average_vector(embeddings_list))
    human_df['average_embeddings'] = average_embeddings_list

    #human_df.to_json('/data/emb_processed_human_more10_lastname_instances_sentences.jsonl', orient='records', force_ascii=False, lines=True)
    ## Error：OverflowError: Maximum recursion level reached

print("クラスタ内のベクトル同士の距離の平均")
print("L1 ")
capital_ijou_cnt = 0
bank_ijou_cnt = 0
pairwise_capital_L1 = 310.33938068547525
pairwise_bank_L1    = 288.0361706233638
for target_ne, target_token_embeddings_list, sentence_count	 in zip(human_df['target_ne'], human_df['target_token_embeddings_list'], human_df['sentence_count']):
    result = cal_group_pairwise_Lp_dist_mean(target_token_embeddings_list, p=1)
    print(f'{target_ne}, {result}, {sentence_count}')
    if pairwise_capital_L1 < result:
        capital_ijou_cnt += 1
    if pairwise_bank_L1 < result:
        bank_ijou_cnt += 1
print(f'capital, {pairwise_capital_L1}, 10')
print(f'bank, {pairwise_bank_L1}, 10')
print(f"{len(human_df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
print(f"{len(human_df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")
print("\n\n")

print("L2 (2乗)")
capital_ijou_cnt = 0
bank_ijou_cnt = 0
pairwise_capital_L2 = 196.15094833155018
pairwise_bank_L2    = 171.06298105715894
for target_ne, target_token_embeddings_list, sentence_count	 in zip(human_df['target_ne'], human_df['target_token_embeddings_list'], human_df['sentence_count']):
    result = cal_group_pairwise_Lp_dist_mean(target_token_embeddings_list, p=2)
    print(f'{target_ne}, {result}, {sentence_count}')
    if pairwise_capital_L2 < result:
        capital_ijou_cnt += 1
    if pairwise_bank_L2 < result:
        bank_ijou_cnt += 1
print(f'capital, {pairwise_capital_L2}, 10')
print(f'bank, {pairwise_bank_L2}, 10')
print(f"{len(human_df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
print(f"{len(human_df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")


print("クラスタ内の平均ベクトルと各ベクトル間の距離の平均")
print("L1 ")
capital_ijou_cnt = 0
bank_ijou_cnt = 0
avevec_capital_L1 = 211.02278376019768
avevec_bank_L1    = 196.26752748101723
for target_ne, target_token_embeddings_list, average_embeddings, sentence_count	 in zip(human_df['target_ne'], human_df['target_token_embeddings_list'], human_df['average_embeddings'], human_df['sentence_count']):
    result = cal_group_avevec_Lp_dist_mean(target_token_embeddings_list, average_embeddings, p=1)
    print(f'{target_ne}, {result}, {sentence_count}')
    if avevec_capital_L1 < result:
        capital_ijou_cnt += 1
    if avevec_bank_L1 < result:
        bank_ijou_cnt += 1
print(f'capital, {avevec_capital_L1}, 10')
print(f'bank, {avevec_bank_L1}, 10')
print(f"{len(human_df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
print(f"{len(human_df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")
print("\n\n")

print("L2 (2乗)")
capital_ijou_cnt = 0
bank_ijou_cnt = 0
avevec_capital_L2 = 91.01377147239918
avevec_bank_L2    = 79.54951687964392
for target_ne, target_token_embeddings_list, average_embeddings, sentence_count	 in zip(human_df['target_ne'], human_df['target_token_embeddings_list'], human_df['average_embeddings'], human_df['sentence_count']):
    result = cal_group_avevec_Lp_dist_mean(target_token_embeddings_list, average_embeddings, p=2)
    print(f'{target_ne}, {result}, {sentence_count}')
    if avevec_capital_L2 < result:
        capital_ijou_cnt += 1
    if avevec_bank_L2 < result:
        bank_ijou_cnt += 1
print(f'capital, {avevec_capital_L2}, 10')
print(f'bank, {avevec_bank_L2}, 10')
print(f"{len(human_df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
print(f"{len(human_df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")

## bokeh argument 
bokeh_target_token_embeddings = []
colors = []
classes = []
label_texts = []
colors_category_list = Category20[20]
cnt = 0
for target_ne, sentence_list, emb, sentence_count	in zip(human_df['target_ne'], human_df['sentence_list'], human_df['target_token_embeddings_list'], human_df['sentence_count']):
    bokeh_target_token_embeddings.extend(np.array(emb))
    classes += ([target_ne] * sentence_count)
    colors += ([colors_category_list[cnt]] * sentence_count)
    for sentence_dict in sentence__list:
        label_texts.append(sentence_dict['sentence'])
        #label_texts.extend(sentence_dict['sentence'])
    cnt += 1
    if cnt >= 20 :
        break

print()
print('print(np.array(bokeh_target_token_embeddings).shape)')
print(np.array(bokeh_target_token_embeddings).shape)
print('print(len(bokeh_target_token_embeddings))')
print(len(bokeh_target_token_embeddings))
print('print(len(bokeh_target_token_embeddings[0]))')
print(len(bokeh_target_token_embeddings[0]))
print('print(type(bokeh_target_token_embeddings))')
print(type(bokeh_target_token_embeddings))
print('print(type(bokeh_target_token_embeddings[0]))')
print(type(bokeh_target_token_embeddings[0]))
print()
print('print(len(colors))')
print(len(colors))
print('print(len(classes))')
print(len(classes))
print('print(len(label_texts))')
print(len(label_texts))
print('print(len(label_texts[0]))')
print(len(label_texts[0]))
print()

## plot embeddings
#plot_embeddings_bokeh(bokeh_target_token_embeddings, emb_method="UMAP", labels=label_texts, classes=classes, color=colors, size=8)
# labelの数が合わずにエラーとなるので消した
plot_embeddings_bokeh(bokeh_target_token_embeddings, emb_method="UMAP", classes=classes, color=colors, size=8)

##plot_embeddings_bokeh(target_token_embeddings, emb_method="UMAP",  labels=label_texts, classes=classes, color=colors, size=20)
## BokehUserWarning: ColumnDataSource's columns must be of the same length. Current lengths: ('class', 1546), ('color', 1546), ('label', 45940), ('x', 1546), ('y', 1546)
## [[sentence],[sentence]]の形にすると良さそう
## appendだと('label', 240)になる．extendだと('label', 45940)になる


## plot multivariate normal distribution
#print('多変量正規分布')
#print('分散共分散行列の対角和')
#
#capital_ijou_cnt = 0
#bank_ijou_cnt = 0
#covariances_trace_capital = 91.38672402011633
#covariances_trace_bank    = 80.12135733510718
#for target_ne, target_token_embeddings_list, sentence_count	 in zip(human_df['target_ne'], human_df['target_token_embeddings_list'], human_df['sentence_count']):
#    result = cal_mnd_covariances_trace(target_token_embeddings_list)
#    print(f'{target_ne} context(×{sentence_count}): {result}')
#    if covariances_trace_capital < result:
#        capital_ijou_cnt += 1
#    if covariances_trace_bank < result:
#        bank_ijou_cnt += 1
#print(f"{len(human_df) - capital_ijou_cnt} < capital < {capital_ijou_cnt}")
#print(f"{len(human_df) - bank_ijou_cnt} < bank < {bank_ijou_cnt}")
#print("\n\n")
#