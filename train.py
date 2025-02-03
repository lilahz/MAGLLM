import time
import argparse
import os
import pickle
import wandb

import torch
import torch.nn.functional as F
import numpy as np

from collections import defaultdict

import constants as c
from model.MAGLLM_lp import MAGLLM_lp
from utils.data import load_ciao_data, get_metapaths_info
from utils.pytorchtools import EarlyStopping
from utils.tools import index_generator, parse_minibatch_Ciao


def run_model_Ciao(category, signal, num_epochs, patience, batch_size, **kwargs):
    dropout_rate = kwargs['dropout_rate']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    profile_config = kwargs['profile_config']
    save_postfix = kwargs['run_name']
    
    (
        adjlists_ur,
        edge_metapath_indices_list_ur,
        _,
        type_mask,
        train_val_test_pos_user_review,
        train_val_test_neg_user_review,
        num_user,
        num_review,
        features,
    ) = load_ciao_data(category, signal, profile_config)

            
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features_type).to(device) for features_type in features.values()]
    in_dims = [features.shape[1] for features in features_list]
    
    train_pos_user_review = train_val_test_pos_user_review['train_pos_user_review']
    val_pos_user_review = train_val_test_pos_user_review['val_pos_user_review']
    test_pos_user_review = train_val_test_pos_user_review['test_pos_user_review']
    train_neg_user_review = train_val_test_neg_user_review['train_neg_user_review']
    val_neg_user_review = train_val_test_neg_user_review['val_neg_user_review']
    test_neg_user_review = train_val_test_neg_user_review['test_neg_user_review']
    
    etypes_lists, num_metapaths_list, num_edge_type, use_masks, no_masks = get_metapaths_info(signal)
    
    for _ in range(1):
        net = MAGLLM_lp(num_metapaths_list, num_edge_type, etypes_lists, in_dims, 64, 64, 8, 128, 'RotatE0', dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        
        if kwargs['use_pretrained']:
            print('Loading pre-saved version of model...')
            net.load_state_dict(torch.load(os.path.join(c.CHECKPOINT_PATH, f'checkpoint_{save_postfix}.pt')))

        # training loop
        print(f'DEBUG: starting train on {len(train_pos_user_review)} samples.')
        net.train()
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, save_path=os.path.join(c.CHECKPOINT_PATH, f'checkpoint_{save_postfix}.pt')
        )
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_user_review))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_review), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_user_review_batch = train_pos_user_review[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_user_review), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_user_review_batch = train_neg_user_review[train_neg_idx_batch].tolist()
                
                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, train_pos_user_review_batch, device, 100, use_masks, num_user)
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, train_neg_user_review_batch, device, 100, no_masks, num_user)
                
                t1 = time.time()
                dur1.append(t1 - t0)

                [pos_embedding_user, pos_embedding_review], _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_review], _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))
                
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_user, pos_embedding_review)
                neg_out = -torch.bmm(neg_embedding_user, neg_embedding_review)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    wandb.log({'train_loss': train_loss.item()})
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_user_review_batch = val_pos_user_review[val_idx_batch].tolist()
                    val_neg_user_review_batch = val_neg_user_review[val_idx_batch].tolist()
                    
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                        adjlists_ur, edge_metapath_indices_list_ur, val_pos_user_review_batch, device, 100, no_masks, num_user)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                        adjlists_ur, edge_metapath_indices_list_ur, val_neg_user_review_batch, device, 100, no_masks, num_user)

                    [pos_embedding_user, pos_embedding_review], _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    [neg_embedding_user, neg_embedding_review], _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    
                    pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_review)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_review)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                    
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            wandb.log({'val_loss': val_loss.item()})
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
            
        pos_test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_user_review), shuffle=False)
        neg_test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_neg_user_review), shuffle=False)
        net.load_state_dict(torch.load(os.path.join(c.CHECKPOINT_PATH, f'checkpoint_{save_postfix}.pt')))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        users_embeddings = defaultdict(list)
        reviews_embeddings = defaultdict(list)
        with torch.no_grad():
            for iteration in range(pos_test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = pos_test_idx_generator.next()
                test_pos_user_review_batch = test_pos_user_review[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, test_pos_user_review_batch, device, 100, no_masks, num_user)

                [pos_embedding_user, pos_embedding_review], _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                
                for i, (user_id, review_id) in enumerate(test_pos_user_review_batch):
                    users_embeddings[user_id].append(pos_embedding_user[i])
                    reviews_embeddings[review_id].append(pos_embedding_review[i])
                
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_review).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                
            for iteration in range(neg_test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = neg_test_idx_generator.next()
                test_neg_user_review_batch = test_neg_user_review[test_idx_batch].tolist()
                
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, test_neg_user_review_batch, device, 100, no_masks, num_user)
                
                [neg_embedding_user, neg_embedding_review], _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                
                for i, (user_id, review_id) in enumerate(test_neg_user_review_batch):
                    users_embeddings[user_id].append(neg_embedding_user[i])
                    reviews_embeddings[review_id].append(neg_embedding_review[i])
                    
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
                
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_review).flatten()
                neg_proba_list.append(torch.sigmoid(neg_out))
                
            users_embeddings = {user_id: torch.mean(torch.stack(embeddings), dim=0, keepdim=True).cpu().numpy() for user_id, embeddings in users_embeddings.items()}
            reviews_embeddings = {review_id: torch.mean(torch.stack(embeddings), dim=0, keepdim=True).cpu().numpy() for review_id, embeddings in reviews_embeddings.items()}
                
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
            with open(os.path.join(c.RESULTS_PATH, f'Ciao{category.replace(" & ", "_")}/y_proba_test_{save_postfix}.npy'), 'wb') as f:
                np.save(f, y_proba_test)

if __name__ == '__main__':
    print('Starting MAGLLM training for Ciao...')
    ap = argparse.ArgumentParser(description='MAGLLM training for the recommendation dataset')
    ap.add_argument('--category', default='Beauty', help='Name of Ciao category. Default is Beauty.')
    ap.add_argument('--signal', default='vote', help='The interaction between the user and the review. Default is vote')
    ap.add_argument('--epoch', type=int, default=10, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=2, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--use-pretrained', type=bool, default=False, help='Use a pre-saved model state. Default is False.')
    
    ap.add_argument('--llm_his_size', type=int, default=10, help='Size of history used to build LLM-based profiles')
    ap.add_argument('--llm_rev_len', type=int, default=150, help='Length of review used to build LLM-based profiles')
    ap.add_argument('--llm_his_order', default='random', help='Order of reviews used to build LLM-based profiles')
    ap.add_argument('--llm_shot', default='zero_shot', help='Type of learning used to build LLM-based profiles')
    ap.add_argument('--llm_reason', default=None, help='Whether use a reasoning in prompt')
    
    
    
    args = ap.parse_args()
    
    profile_config = f'his_size_{args.llm_his_size}_rev_len_{args.llm_rev_len}_order_{args.llm_his_order}_{args.llm_shot}'
    if args.llm_reason is not None:
        profile_config = f'{profile_config}_reason_{args.llm_reason}'
    
    for lr in [0.0001, 0.0005]:
        for weight_decay in [0, 0.0001, 0.0005, 0.00001, 0.00005]:
            for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for batch_size in [16, 32, 64, 128]:
                
                    run_name = f'Ciao{args.category}_{args.signal}_w2v_profiles_{lr}_{weight_decay}_{dropout_rate}_{batch_size}'
                    run_name = f'{run_name}_{profile_config}'
                    
                    wandb.init(
                        # set the wandb project where this run will be logged
                        project='personalized_reviews',
                        name=run_name,
                        
                        # track hyperparameters and run metadata
                        config = {
                            'learning_rate': lr,
                            'weight_decay': weight_decay, 
                            'dropout_rate': dropout_rate, 
                            'batch_size': batch_size,
                            'model_name': f'Ciao{args.category}',
                            'llm_his_size': args.llm_his_size,
                            'llm_rev_len': args.llm_rev_len,
                            'llm_his_order': args.llm_his_order,
                            'llm_shot': args.llm_shot
                        }
                    )
                    
                    run_model_Ciao(args.category, args.signal, args.epoch, args.patience, args.batch_size, 
                                    lr=lr, weight_decay=weight_decay, dropout_rate=dropout_rate, run_name=run_name, 
                                    use_pretrained=args.use_pretrained, profile_config=profile_config, llm_his_size=args.llm_his_size, 
                                    llm_rev_len=args.llm_rev_len, llm_his_order=args.llm_his_order, llm_shot=args.llm_shot)
