import os
import sys

import numpy as np
import scipy
import pandas as pd
import pickle
import time
import torch

from constants import DATA_PATH


def load_ciao_data(category, signal, profile_config = ''):
    print('Loading data...')
    
    start_time = time.time()
    path = os.path.join(DATA_PATH, f'Ciao{category.replace(" & ", "_")}_{signal}')
    
    adj_lists = []
    index_lists = []
    for root, dirs, files in os.walk(path):
        for i, dir in enumerate(dirs):
            adj_lists.append([])
            adj_files = sorted([f for f in os.listdir(os.path.join(path, dir)) if f.endswith('.adjlist')])
            for file in adj_files:
                with open(os.path.join(path, dir, file), 'r') as f:
                    adjlist = [line.strip() for line in f]
                    adj_lists[i].append(adjlist)
                
            index_lists.append([])
            index_files = sorted([f for f in os.listdir(os.path.join(path, dir)) if f.endswith('.pickle')])
            for file in index_files:
                with open(os.path.join(path, dir, file), 'rb') as f:
                    idx = pickle.load(f)
                    index_lists[i].append(idx)
                    
    adjM = scipy.sparse.load_npz(os.path.join(path, 'adj_mat.npz'))
    type_mask = np.load(os.path.join(path, 'node_types.npy'))
    train_val_test_pos_user_review = np.load(os.path.join(path, 'train_val_test_pos_user_review.npz'))
    train_val_test_neg_user_review = np.load(os.path.join(path, 'train_val_test_neg_user_review.npz'))
    
    with open(os.path.join(path, 'mappings.pickle'), 'rb') as f:
        mapping = pickle.load(f)
        
        num_user = len(mapping['umap'])
        num_review = len(mapping['rmap'])
    
    features = np.load(os.path.join(path, f'w2v_profiles_implicit_features_{profile_config}.npz'))
        
    print(f'Data loading finished after {time.time() - start_time:.2f} seconds')

    return adj_lists, index_lists, adjM, type_mask, train_val_test_pos_user_review, \
            train_val_test_neg_user_review, num_user, num_review, features
            
def get_metapaths_info(signal):
    if signal == 'like':
        # 0: user liked review, 1: review liked by user, 2: review under product
        # 3: product contains review, 4: user liked review under product, 5: product contains review liked by user
        etypes_lists = [[[0, 1], [0, 2, 3, 1], [4, 5]],
                        [[1, 0], [2, 5, 4, 3], [2, 3]]]
        num_metapaths_list = [len(lst) for lst in etypes_lists]
        num_edge_type = 6
        use_masks = [[True, True, False],
                    [True, True, False]]
        no_masks = [[False] * mp for mp in num_metapaths_list]
    elif signal == 'write':
        # 0: user wrote review, 1: review written by user, 2: review under product
        # 3: product contains review, 4: user wrote review under product, 5: product contains review written by user
        etypes_lists = [[[0, 2, 3, 1], [4, 5]],
                        [[1, 0], [2, 5, 4, 3], [2, 3]]]
        num_metapaths_list = [len(lst) for lst in etypes_lists]
        num_edge_type = 6
        use_masks = [[True, False],
                    [True, True, False]]
        no_masks = [[False] * mp for mp in num_metapaths_list]
    elif signal == 'both':
        # 0: user liked review, 1: review liked by user, 2: review under product, 3: product contains review
        # 4: user liked review under product, 5: product contains review liked by user
        # 6: user written review, 7: review written by user
        # 8: user wrote review under product, 9: product contains review written by user
        etypes_lists = [[[0, 1], [0, 2, 3, 1], [4, 5], [6, 2, 3, 7], [8, 9]],
                        [[1, 0], [2, 5, 4, 3], [2, 3], [2, 9, 8, 3], [7, 6]]]
        num_metapaths_list = [len(lst) for lst in etypes_lists]
        num_edge_type = 10
        use_masks = [[True, True, True, False, False],
                    [True, True, False, False, False]]
        no_masks = [[False] * mp for mp in num_metapaths_list]
    else:
        raise Exception('wrong signal was given.')
        
    return etypes_lists, num_metapaths_list, num_edge_type, use_masks, no_masks
