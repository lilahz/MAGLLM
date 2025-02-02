import argparse
import numpy as np
import pickle
import random
import scipy.sparse

from gensim.models import KeyedVectors
from tqdm import tqdm

import constants as c
from utils.utils import * 
from utils.metapaths import * 

tqdm.pandas()

global COMMON
COMMON = {}

def map_entities(votes, reviews):
    users = list(votes['voter_id'].unique())
    reviews = reviews['review_id'].tolist()
    products = list(votes['product_id'].unique())
    
    COMMON['num_users'] = len(users)
    COMMON['num_reviews'] = len(reviews)
    COMMON['num_products'] = len(products)
    
    users_id2idx, users_idx2id = node_mapping(users)
    reviews_id2idx, reviews_idx2id = node_mapping(reviews)
    products_id2idx, products_idx2id = node_mapping(products)
    
    mappings = {
        'umap': users_idx2id,
        'rmap': reviews_idx2id,
        'pmap': products_idx2id
    }

    with open(os.path.join(COMMON['output_path'], 'mappings.pickle'), 'wb') as f:
        pickle.dump(mappings, f)
        
    id_mapping = {
        'users': users_id2idx,
        'reviews': reviews_id2idx,
        'products': products_id2idx
    }
    
    return id_mapping

def split_to_pos_neg_pairs(signal, votes, user_review, author_review, split_map):
    pairs = {
        'train': {'positive': [], 'negative': []}, 
        'valid': {'positive': [], 'negative': []}, 
        'test': {'positive': [], 'negative': []}
    }
    
    for split in ['train', 'valid', 'test']:
        if signal == 'write' and split == 'train':
            pos_candidates = set(map(tuple, author_review))
        elif signal == 'both' and split == 'train':
            pos_candidates = np.concatenate([user_review[split_map['train']], np.array(author_review).astype(int)])
            pos_candidates = set(map(tuple, pos_candidates))
        else: 
            pos_candidates = set(map(tuple, user_review[split_map[split]]))
        
        if split == 'train':
            df_for_negs = votes[votes['split'] == 'train'].groupby('product_idx', as_index=False).agg({'voter_idx': set, 'review_idx': set})
            for _, row in tqdm(df_for_negs.iterrows(), total=len(df_for_negs)):
                product_reviews = list(row['review_idx'])
                
                for vid in row['voter_idx']:
                    pos = [(vid, rid) for rid in product_reviews if (vid, rid) in pos_candidates]
                    neg = [(vid, rid) for rid in product_reviews if (vid, rid) not in pos_candidates]
                    
                    for single_pos in pos:
                        if neg:
                            pairs['train']['positive'].append(single_pos)
                            pairs['train']['negative'].append(random.choice(neg))
        else:
            df_for_negs = votes[votes['split'] == split].groupby('product_idx', as_index=False).agg({'voter_idx': set, 'review_idx': set})
            for _, row in tqdm(df_for_negs.iterrows(), total=len(df_for_negs)):
                product_reviews = list(row['review_idx'])
                
                for vid in row['voter_idx']:
                    pos = [(vid, rid) for rid in product_reviews if (vid, rid) in pos_candidates]
                    neg = [(vid, rid) for rid in product_reviews if (vid, rid) not in pos_candidates]
                    
                    pairs[split]['positive'].extend(pos)
                    pairs[split]['negative'].extend(neg)
        
    np.savez(os.path.join(COMMON['output_path'], 'train_val_test_neg_user_review.npz'),
             train_neg_user_review=pairs['train']['negative'],
             val_neg_user_review=pairs['valid']['negative'],
             test_neg_user_review=pairs['test']['negative'])
    np.savez(os.path.join(COMMON['output_path'], 'train_val_test_pos_user_review.npz'),
             train_pos_user_review=pairs['train']['positive'],
             val_pos_user_review=pairs['valid']['positive'],
             test_pos_user_review=pairs['test']['positive'])
                
    return pairs['train']['positive']

def build_adj_matrix_and_metapaths(signal, user_review, author_review, review_product, user_product, author_product):
    # build the adjacency matrix
    # 0 for user, 1 for review, 2 for product
    num_users, num_reviews, num_products = COMMON['num_users'], COMMON['num_reviews'], COMMON['num_products']
    dim = num_users + num_reviews + num_products

    type_mask = np.zeros((dim), dtype=int)
    type_mask[num_users:num_users+num_reviews] = 1
    type_mask[num_users+num_reviews:] = 2

    adj_mat = np.zeros((dim, dim), dtype=int)
    for _, row in tqdm(user_review.iterrows(), total=len(user_review)):
        uid = row['user_idx']
        rid = num_users + row['review_idx']
        adj_mat[uid, rid] = 1
        adj_mat[rid, uid] = 1
    if signal == 'both':
        for _, row in tqdm(author_review.iterrows(), total=len(author_review)):
            aid = row['author_idx']
            rid = num_users + row['review_idx']
            adj_mat[aid, rid] = 2
            adj_mat[rid, aid] = 2
    for _, row in tqdm(review_product.iterrows(), total=len(review_product)):
        rid = num_users + row['review_idx']
        pid = num_users + num_reviews + row['product_idx']
        adj_mat[rid, pid] = 1
        adj_mat[pid, rid] = 1
    for _, row in tqdm(user_product.iterrows(), total=len(user_product)):
        uid = row['user_idx']
        pid = num_users + num_reviews + row['product_idx']
        adj_mat[uid, pid] = 1
        adj_mat[pid, uid] = 1
    if signal == 'both':
        for _, row in tqdm(author_product.iterrows(), total=len(author_product)):
            uid = row['author_idx']
            pid = num_users + num_reviews + row['product_idx']
            adj_mat[uid, pid] = 3
            adj_mat[pid, uid] = 3
            
    user_review_list = {i: adj_mat[i, num_users:num_users+num_reviews].nonzero()[0] for i in range(num_users)}
    review_user_list = {i: adj_mat[num_users + i, :num_users].nonzero()[0] for i in range(num_reviews)}
    review_product_list = {i: adj_mat[num_users + i, num_users+num_reviews:].nonzero()[0] for i in range(num_reviews)}
    product_review_list = {i: adj_mat[num_users + num_reviews + i, num_users:num_users+num_reviews].nonzero()[0] for i in range(num_products)}
    user_product_list = {i: adj_mat[i, num_users+num_reviews:].nonzero()[0] for i in range(num_users)}
    product_user_list = {i: adj_mat[num_users+num_reviews+i, :num_users].nonzero()[0] for i in range(num_products)}

    if signal == 'both':
        author_review_list = {i: np.where(adj_mat[i, num_users:num_users+num_reviews] == 2)[0] for i in range(num_users)}
        review_author_list = {i: np.where(adj_mat[num_users + i, :num_users] == 2)[0] for i in range(num_reviews)}
        author_product_list = {i: np.where(adj_mat[i, num_users+num_reviews:] == 3)[0] for i in range(num_users)}
        product_author_list = {i: np.where(adj_mat[num_users+num_reviews+i, :num_users] == 3)[0] for i in range(num_products)}
    else:
        author_review_list = {}
        review_author_list = {}
        author_product_list = {}
        product_author_list = {}
        
    expected_metapaths, metapaths_functions = get_metapaths(signal, COMMON['output_path'])
    # write all things
    target_idx_lists = [np.arange(num_users), np.arange(num_reviews), np.arange(num_products)]
    offset_list = [0, num_users, num_products]
    for i, metapaths in enumerate(expected_metapaths):
        for metapath in metapaths:
            edge_metapath_idx_array = metapaths_functions[metapath](
                user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
                product_user_list, product_review_list, author_product_list, product_author_list, 
                num_users, num_reviews
            )
            
            with open(COMMON['output_path'] + '/' + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
                target_metapaths_mapping = {}
                left = 0
                right = 0
                for target_idx in target_idx_lists[i]:
                    while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                        right += 1
                    target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                    left = right
                pickle.dump(target_metapaths_mapping, out_file)
            
            with open(COMMON['output_path'] + '/' + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
                left = 0
                right = 0
                for target_idx in target_idx_lists[i]:
                    while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                        right += 1
                    neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
                    neighbors = list(map(str, neighbors))
                    if len(neighbors) > 0:
                        out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                    else:
                        out_file.write('{}\n'.format(target_idx))
                    left = right

    scipy.sparse.save_npz(COMMON['output_path'] + '/adj_mat.npz', scipy.sparse.csr_matrix(adj_mat))
    np.save(COMMON['output_path'] + '/node_types.npy', type_mask)
        
def build_interactions(signal, votes, reviews):
    user_review, author_review, review_product, user_product, author_product = [], [], set(), set(), set()
    idx_dict = {'train': [], 'valid': [], 'test': []}
    positive_vote = [3, 4, 5]
    
    count = 0
    for idx, row in tqdm(votes.iterrows(), total=len(votes)):
        voter_idx = row['voter_idx']
        review_idx = row['review_idx']
        product_idx = row['product_idx']
        vote = row['vote']
        
        review_product.add((review_idx, product_idx))
        
        if vote in positive_vote:
            split = row['split']
            
            if split == 'train' and signal in ['vote', 'both']:
                user_product.add((voter_idx, product_idx))
            
            idx_dict[split].append(count)
            count += 1
            
            user_review.append((voter_idx, review_idx))

    for idx, row in tqdm(reviews.iterrows(), total=len(reviews)):
        user_idx = row['user_idx']
        review_idx = row['review_idx']
        product_idx = row['product_idx']
        
        review_product.add((review_idx, product_idx))
        
        if row['user_id'] not in votes['voter_id'].tolist():
            continue
        
        author_review.append((user_idx, review_idx))
        author_product.add((user_idx, product_idx))
        
    user_review = np.array(user_review).astype(int)
    author_review = np.array(author_review).astype(int)
    review_product = list(review_product)
    user_product = list(user_product)

    user_review = split_to_pos_neg_pairs(signal, votes, user_review, author_review, idx_dict)
    
    user_review = pd.DataFrame(user_review, columns=['user_idx', 'review_idx'])
    author_review = pd.DataFrame(author_review, columns=['author_idx', 'review_idx'])
    review_product = pd.DataFrame(review_product, columns=['review_idx', 'product_idx'])
    user_product = pd.DataFrame(user_product, columns=['user_idx', 'product_idx'])
    author_product = pd.DataFrame(author_product, columns=['author_idx', 'product_idx'])
    
    build_adj_matrix_and_metapaths(signal, user_review, author_review, review_product, user_product, author_product)
    
def build_features(profiles, reviews):
    with open(os.path.join(COMMON['output_path'], 'mappings.pickle'), 'rb') as f:
        mappings = pickle.load(f)
        
    w2v_model = KeyedVectors.load(c.W2V_MODEL_PATH)
    
    reviews['tokens'] = reviews['clean_review'].progress_apply(tokenize_text)
    profiles['tokens'] = profiles['profile'].progress_apply(tokenize_text)
    
    reviews['embedding'] = reviews['tokens'].progress_apply(lambda t: w2v_vectorize(w2v_model, t))
    profiles['embedding'] = profiles['tokens'].progress_apply(lambda t: w2v_vectorize(w2v_model, t))
    
    user_features = []
    for user_idx, user_id in tqdm(mappings['umap'].items()):
        try:
            embeddings = profiles[profiles['user_id'] == user_id]['embedding'].values[0]
        except:
            embeddings = np.zeros((300, ))
        
        user_features.append(embeddings)
    
    review_features = []
    for review_idx, review_id in tqdm(mappings['rmap'].items()):
        embedding = reviews[reviews['review_id'] == review_id]['embedding'].values[0]
        review_features.append(embedding)
        
    product_features = []
    for product_idx, product_id in tqdm(mappings['pmap'].items()):
        embeddings = np.array(reviews[reviews['product_id'] == product_id]['embedding'].tolist())
        product_features.append(np.mean(embeddings, axis=0))
        
    np.savez(os.path.join(COMMON['output_path'], f'w2v_profiles_implicit_features_{profile_config}.npz'),
         user_features=user_features,
         review_features=review_features,
         product_features=product_features)

if __name__ == '__main__':
    print('Starting Ciao preprocess for MAGLLM training')
    ap = argparse.ArgumentParser(description='MAGNN testing for the recommendation dataset')
    ap.add_argument('--category', default='Beauty', help='Name of Ciao category. Default is Beauty.')
    ap.add_argument('--signal', default='vote', help='The interaction between the user and the review. Default is vote')
    ap.add_argument('--llm_his_size', type=int, default=10, help='Size of history used to build LLM-based profiles')
    ap.add_argument('--llm_rev_len', type=int, default=150, help='Length of review used to build LLM-based profiles')
    ap.add_argument('--llm_his_order', default='random', help='Order of reviews used to build LLM-based profiles')
    ap.add_argument('--llm_shot', default='zero_shot', help='Type of learning used to build LLM-based profiles')
    ap.add_argument('--llm_reason', default=None, help='Whether use a reasoning in prompt')
    
    args = ap.parse_args()
    
    output_path = os.path.join(c.DATA_PATH, f'Ciao{args.category.replace(" & ", "_")}_{args.signal}')
    COMMON['output_path'] = output_path
    os.makedirs(output_path, exist_ok=True)

    votes, reviews = read_data_files(args.category)
    
    id_mapping = map_entities(votes, reviews)
    
    votes['voter_idx'] = votes['voter_id'].map(id_mapping['users'])
    votes['review_idx'] = votes['review_id'].map(id_mapping['reviews'])
    votes['product_idx'] = votes['product_id'].map(id_mapping['products'])
    reviews['user_idx'] = reviews['user_id'].map(id_mapping['users'])
    reviews['review_idx'] = reviews['review_id'].map(id_mapping['reviews'])
    reviews['product_idx'] = reviews['product_id'].map(id_mapping['products'])
    
    # build nodes interactions based on signal
    build_interactions(args.signal, votes, reviews)
    
    # features
    profile_config = f'his_size_{args.llm_his_size}_rev_len_{args.llm_rev_len}_order_{args.llm_his_order}_{args.llm_shot}'
    if args.llm_reason:
        profile_config = f'{profile_config}_reason_{args.llm_reason}'
    profiles = pd.read_csv(
        os.path.join(c.PROFILES_PATH, f'ciao_{args.category.replace(" & ", "_")}', profile_config, f'{args.signal}_train_profiles.csv')
    )
    build_features(profiles, reviews)