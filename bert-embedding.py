import argparse
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine
from plot import plot_embeddings_bokeh


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
texts = ["A rock stuck out from the bank into the river.",
         "We walked on the bank of the Thames.",
         "The river flowed over its bank.",
         "The children slid down the bank.",
         "The bank was out of money.",
         "Would you please check this matter with your bank?",
         "He cashed a check at the bank.",
         "I had to wait in a line at the bank."]
colors = ["blue", "blue","blue","blue",
          "red", "red", "red", "red" ]
classes = ["river", "river", "river", "river", 
          "money", "money", "money", "money"]

# Getting embeddings for the target
# word in all given contexts
target_word_embeddings = []
target_word = 'bank'

for text in texts:
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    
    # Find the position 'bank' in list of tokens
    word_index = tokenized_text.index(target_word)
    # Get the embedding for bank
    word_embedding = list_token_embeddings[word_index]

    target_word_embeddings.append(word_embedding)



# Calculating the cosine similarity between the
# embeddings of 'bank' in all the
# given contexts of the word

list_of_cos_similarity = []
for text1, embed1 in zip(texts, target_word_embeddings):
    for text2, embed2 in zip(texts, target_word_embeddings):
        cos_sim = 1 - cosine(embed1, embed2) # 'cosine' is cosine distance 
        list_of_cos_similarity.append([text1, text2, cos_sim])

cos_similarity_df = pd.DataFrame(list_of_cos_similarity, columns=['text1', 'text2', 'cosine similarity'])
print(f'\ntarget word is {target_word}')
#print(cos_similarity_df)

# plot embeddings
target_word_embeddings = np.array(target_word_embeddings)
plot_embeddings_bokeh(target_word_embeddings, emb_method="UMAP",  labels=texts, classes=classes, color=colors, size=20)