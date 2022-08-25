import argparse
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine
from plot import plot_embeddings_bokeh
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from bokeh.palettes import Category10

def plot_mnd(emb, outfile="GMM.png"):
    # concatenate the two datasets into the final training set
    #X_train = np.vstack([shifted_gaussian, stretched_gaussian])
    X_train = emb

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=1, covariance_type="full")
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    #x = np.linspace(-20.0, 30.0)
    #y = np.linspace(-20.0, 40.0)
    #X, Y = np.meshgrid(x, y)
    #XX = np.array([X.ravel(), Y.ravel()]).T
    #Z = -clf.score_samples(XX)
    #Z = Z.reshape(X.shape)
#
    #CS = plt.contour(
    #    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    #)
    #CB = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1])
    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.savefig(outfile)
    plt.show()
    #print(clf.covariances_)
    #print(clf.covariances_.shape)
    print(f'共分散行列の対角和：{np.trace(clf.covariances_[0])}')
    print()

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
    return L2_dist_mean

def cal_group_avevec_L2_dist_mean(embeddings, ave_vector, p=2):
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
    return L2_dist_mean

def generate_average_vector(embeddings):
    """
    embeddings: ndarray.  size of ([n,768]) 
    output : torch.tensor
    """
    embeddings = torch.tensor(embeddings)
    average_vector = torch.sum(embeddings, axis=0)
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
        list: List of list of floats of size
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
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

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

# Text corpus
##############
# These sentences show the different
# forms of the word 'bank' to show the
# value of contextualized embeddings
sentences_dic= {
            "Obama" : 
                [   "Obama was born in Honolulu, Hawaii. ",
                    "Obama signed many landmark bills into law during his first two years in office.",
                    "After winning re-election by defeating Republican opponent Mitt Romney, Obama was sworn in for a second term on January 20, 2013.",
                    "Obama was elected over Republican nominee John McCain in the general election and was inaugurated alongside his running mate Joe Biden, on January 20, 2009. ",
                    "Obama was elected President.",
                    "Then Obama made an inaugural speech for about 20 minutes.",
                    "U.S. President Barack Obama attended the game and threw out the ceremonial first pitch.",
                    "Obama said the American people must remain vigilant.",
                    "President Obama Meets With Myanmar's Aung San Suu Kyi.",
                    "Romney criticized Obama's economic policies during his first term."
                ],
            "Clinton" : 
                [   
                    "Clinton was born and raised in Arkansas and attended Georgetown University. ",
                    "Clinton presided over the longest period of peacetime economic expansion in American history.",
                    "Clinton was elected president in the 1992 presidential election, defeating incumbent Republican president George H. W. Bush and independent businessman Ross Perot. ",
                    "Clinton left office in 2001 with the joint-highest approval rating of any U.S. ",
                    "The Republican Party won unified control of Congress for the first time in 40 years in the 1994 elections, but Clinton was still comfortably re-elected in 1996, becoming the first Democrat since Franklin D. Roosevelt to win a second full term. ",
                    "Clinton was elected President.",
                    "Clinton had asked to meet with the Empress again on this visit and the Empress invited her as a former first lady.",
                    "He defeated Democrat Hillary Clinton, 69, a former First Lady and Secretary of State, by a narrow margin.",
                    "Mr. Clinton was elected by an overwhelming majority.",
                    "Bill Clinton was elected president of the United States."        
                ],
            "dog" : 
                [   
                    "This makes the domestic dog the most popular pet on the planet.",
                    "A third of all households worldwide have a dog, according to a 2016 consumer insights study.",
                    "For instance, he says kids should ask for permission from the dog’s owner before trying to pet or play with the animal. ",
                    "And each time Fido stops to sniff a fire hydrant on your walk, it’s analyzing the pheromones left behind by another dog’s urine.",
                    "Chasing sticks and balls may be linked to the pursuit of prey, while digging at the carpet or a dog bed echoes how a wild canid would prepare its sleeping area.",
                    "Their dog was so fierce that he kept everyone away.",
                    "The dog defended his master from harm.",
                    "The dog barked all night.",
                    "Don't let the dog in.",
                    "There's no dog in the yard."
                ],
            "bank" :
                [
                    "A rock stuck out from the bank into the river.",
                    "We walked on the bank of the Thames.",
                    "The river flowed over its bank.",
                    "The children slid down the bank.",
                    "He stood on the bank, breathing heavily.",
                    "The bank was out of money.",
                    "He cashed a check at the bank.",
                    "I had to wait in a line at the bank.",
                    "Would you please check this matter with your bank?",
                    "The bank transfer has been completed."
                ]
            }

## bokeh argument 
label_texts = sentences_dic.copy()
label_texts = list(itertools.chain.from_iterable(list(label_texts.values())))
colors_category = Category10[len(sentences_dic.keys()) +1]
colors = []
classes = []
for i, key in enumerate(sentences_dic.keys()):
    colors += ([colors_category[i]] * len(sentences_dic[key]))
    classes += ([key] * len(sentences_dic[key]))
for i, key in enumerate(sentences_dic.keys()):
    label_texts.append("average_"+key)
    classes.append("average_"+key)
colors += ([colors_category[len(sentences_dic.keys())]] * len(sentences_dic.keys()))

print(colors)
print(classes)

# Getting embeddings for the target
# word in all given contexts
target_word_embeddings = []
for target_word, sentences in sentences_dic.items():
    for sentence in sentences:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        # Find the position target_word in list of tokens
        word_index = tokenized_text.index(tokenizer.tokenize(target_word)[0])
        # Get the embedding for target_word
        word_embedding = list_token_embeddings[word_index]
        target_word_embeddings.append(word_embedding)

# Calculating the cosine similarity between the
# embeddings of target_word in all the given contexts of the word
#list_of_cos_similarity = []
#for text1, embed1 in zip(sentences_dic, target_word_embeddings):
#    for text2, embed2 in zip(sentences_dic, target_word_embeddings):
#        cos_sim = 1 - cosine(embed1, embed2) # 'cosine' is cosine distance 
#        list_of_cos_similarity.append([text1, text2, cos_sim])
#cos_similarity_df = pd.DataFrame(list_of_cos_similarity, columns=['text1', 'text2', 'cosine similarity'])
#print(cos_similarity_df)

target_word_embeddings = np.array(target_word_embeddings)
## 以下後でFixする(keyの数の分、自動で保存する)
# split embeddings
target_word_embeddings_obama, target_word_embeddings_clinton, target_word_embeddings_dog, target_word_embeddings_bank = np.vsplit(target_word_embeddings, len(sentences_dic.keys()))
splited_target_word_embeddings = np.vsplit(target_word_embeddings, len(sentences_dic.keys()))

# generate average vector
average_embeddings_obama = generate_average_vector(target_word_embeddings_obama)
average_embeddings_clinton = generate_average_vector(target_word_embeddings_clinton)
average_embeddings_dog = generate_average_vector(target_word_embeddings_dog)
average_embeddings_bank = generate_average_vector(target_word_embeddings_bank)

print("各ベクトル間の距離の平均")
print("L1 Norm mean")
print(f'Obama context(×{len(target_word_embeddings_obama)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_obama, p=1)}')
print(f'Clinton context(×{len(target_word_embeddings_clinton)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_clinton, p=1)}')
print(f'dog context(×{len(target_word_embeddings_dog)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_dog, p=1)}')
print(f'bank context(×{len(target_word_embeddings_bank)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_bank, p=1)}\n')
print("L2 Norm mean")
print(f'Obama context(×{len(target_word_embeddings_obama)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_obama, p=2)}')
print(f'Clinton context(×{len(target_word_embeddings_clinton)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_clinton, p=2)}')
print(f'dog context(×{len(target_word_embeddings_dog)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_dog, p=2)}')
print(f'bank context(×{len(target_word_embeddings_bank)}): {cal_group_pairwise_Lp_dist_mean(target_word_embeddings_bank, p=2)}\n')

print("平均ベクトルからの距離の平均")
print("L1 Norm mean")
print(f'Obama context(×{len(target_word_embeddings_obama)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_obama, average_embeddings_obama, p=1)}')
print(f'Clinton context(×{len(target_word_embeddings_clinton)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_clinton, average_embeddings_clinton, p=1)}')
print(f'dog context(×{len(target_word_embeddings_dog)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_dog, average_embeddings_dog, p=1)}')
print(f'bank context(×{len(target_word_embeddings_bank)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_bank, average_embeddings_bank, p=1)}\n')
print("L2 Norm mean")
print(f'Obama context(×{len(target_word_embeddings_obama)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_obama, average_embeddings_obama, p=2)}')
print(f'Clinton context(×{len(target_word_embeddings_clinton)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_clinton, average_embeddings_clinton, p=2)}')
print(f'dog context(×{len(target_word_embeddings_dog)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_dog, average_embeddings_dog, p=2)}')
print(f'bank context(×{len(target_word_embeddings_bank)}): {cal_group_avevec_L2_dist_mean(target_word_embeddings_bank, average_embeddings_bank, p=2)}\n')

# plot embeddings
## add average embeddings
target_word_embeddings = np.vstack((target_word_embeddings, np.array(average_embeddings_obama)))
target_word_embeddings = np.vstack((target_word_embeddings, np.array(average_embeddings_clinton)))
target_word_embeddings = np.vstack((target_word_embeddings, np.array(average_embeddings_dog)))
target_word_embeddings = np.vstack((target_word_embeddings, np.array(average_embeddings_bank)))

plot_embeddings_bokeh(target_word_embeddings, emb_method="UMAP",  labels=label_texts, classes=classes, color=colors, size=20)

# plot multivariate normal distribution
print("obama")
plot_mnd(target_word_embeddings_obama, outfile="mnd_obama")
print("clinton")
plot_mnd(target_word_embeddings_clinton, outfile="mnd_clinton")
print("dog")
plot_mnd(target_word_embeddings_dog, outfile="mnd_dog")
print("bank")
plot_mnd(target_word_embeddings_bank, outfile="mnd_bank")