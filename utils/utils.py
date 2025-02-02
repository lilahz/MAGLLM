import itertools
import numpy as np
import os
import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import constants as c


def read_data_files(category):
    votes = pd.read_csv(os.path.join(c.CSVS_PATH, 'votes_4core_split_60_20_20.tsv'), sep='\t')
    votes = votes[votes['category'] == category]
    
    reviews = pd.read_csv(os.path.join(c.CSVS_PATH, 'ciao_4core.tsv'), sep='\t')
    reviews = reviews[reviews['category'] == category]
    
    return votes, reviews

def node_mapping(nodes):
    # id2idx = mapping betweem the node original id and its sorted index
    # idx2id = mapping between the sorted index and the corresponsing original id
    id2idx, idx2id = {}, {}

    for idx, node in enumerate(nodes):
        id2idx[node] = idx
        idx2id[idx] = node

    return id2idx, idx2id

def split_tokens_with_punctuation(tokens):
    punctuation_pattern = re.compile(r'[^\w\s]|\s{2,}')
    new_tokens = []
    for token in tokens:
        sub_tokens = punctuation_pattern.split(token)
        for sub_token in sub_tokens:
            new_tokens.append(sub_token)
    return new_tokens

def filter_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation) | {'*', '&'}
    alphabetic_pattern = re.compile(r'[a-zA-Z]+')
    tokens = [token.lower() for token in tokens if token not in punctuation and alphabetic_pattern.fullmatch(token)]
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def tokenize_text(text, tokenizer='word'):
    """
    Preprocess and tokenize the text. 

    The preprocessing includes removal of punctuaion marks, numbers and stop words. 
    The return value if list of filtered tokens or sentences. 
    """
    # Remove words with repeating characters
    repeat_pattern = re.compile(r'([^\W\d_])\1{2,}')
    text = re.sub(repeat_pattern, lambda x: x.group()[0] * 2, text)
    
    # Tokenize and split tokens with punctuation
    sentences = sent_tokenize(text)
    sentences = [word_tokenize(sentence) for sentence in sentences]
    sentences = [split_tokens_with_punctuation(tokens) for tokens in sentences]

    # Remove stop words, punctuation and non-alphabetic characters
    sentences = [filter_tokens(tokens) for tokens in sentences]
    
    # Return wanted structure based on tokenizer parameter
    if tokenizer == 'word':
        return list(itertools.chain(*sentences))
    else:
        return [' '.join(sentence) for sentence in sentences]
    
def w2v_vectorize(w2v_model, words):
    vectors = []
    
    for word in words:
        if len(word) == 1:
            continue
        try:    
            vectors.append(w2v_model[word])
        except:
            continue
    
    return np.average(vectors, axis=0)