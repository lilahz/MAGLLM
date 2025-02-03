import argparse
import os
import pandas as pd
import re
import random

from langchain.llms import LlamaCpp
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import constants as c
from utils.utils import read_data_files


def cut_review(review, review_len):
    sentences = sent_tokenize(' '.join(review.split()[:review_len]))
    last_sentence = sentences[-1]
    if len(last_sentence.split()) < 10:
        sentences = sentences[:-1]
    
    return ' '.join(sentences)

def load_data(category, review_len):
    votes, reviews = read_data_files(category)
    
    users = list(votes[votes['split'] == 'train']['voter_id'].unique())
    
    written_reviews = {}

    for _, row in tqdm(reviews.iterrows(), total=len(reviews)):
        user_id = row['user_id']

        if user_id not in written_reviews:
            written_reviews[user_id] = []
        written_reviews[user_id].append((row['review_id'], row['category'], row['product']))

    voted_reviews = {}
    positive_vote = [3, 4, 5]

    for _, row in tqdm(votes.iterrows(), total=len(votes)):
        voter_id = row['voter_id']
        split = row['split']
        
        if row['vote'] in positive_vote:
            if voter_id not in voted_reviews:
                voted_reviews[voter_id] = {}
            if split not in voted_reviews[voter_id]:
                voted_reviews[voter_id][split] = []
            voted_reviews[voter_id][split].append((row['review_id'], row['vote'], row['category'], row['product']))
            
    reviews['review_concat'] = reviews['clean_review'].apply(lambda t: cut_review(t, review_len))
    review_id_text_mapping = dict(zip(reviews.review_id, reviews.review_concat))
            
    return users, written_reviews, voted_reviews, review_id_text_mapping

def load_history(voter_id, rid2index, history_size, his_order, written_review, voted_reviews, history_type, category):
    history_write = [(i[1], i[2], rid2index[i[0]]) for i in written_review[voter_id]]
    
    if 'train' in voted_reviews.get(voter_id, []):
        voted_reviews_user = voted_reviews[voter_id]['train']
    else:
        voted_reviews_user = []
        
    history_vote = [(i[2], i[3], rid2index[i[0]]) for i in voted_reviews_user if i[1] != category]

    if history_type == 'write':
        history = random.sample(history_write, min(history_size, len(history_write)))
    elif history_type == 'vote':
        if his_order == 'random':
            history = random.sample(history_vote, min(history_size, len(history_vote)))
        elif his_order == 'rel_first':
            voted_reviews_user = sorted(voted_reviews_user, key=lambda x:x[1], reverse=True)
            history_vote = [(i[2], i[3], rid2index[i[0]]) for i in voted_reviews_user if i[1] != category]
            
            history = history_vote[:history_size]
        elif his_order == 'rel_last':
            voted_reviews_user = sorted(voted_reviews_user, key=lambda x:x[1])
            history_vote = [(i[2], i[3], rid2index[i[0]]) for i in voted_reviews_user if i[1] != category]
            
            history = history_vote[:history_size]
    else:  # 'both'
        history = (list(random.sample(history_write, min(history_size//2, len(history_write))) +
                   list(random.sample(history_vote, min(history_size//2, len(history_vote))))))

    return history

def build_profile(llm, signal, history, shot='zero', reasoning=None):
    if signal == 'vote':
        past_signal = 'voted'
    elif signal == 'write':
        past_signal = 'written'
    else:
        past_signal = 'voted or wrote'
        
    if shot == 'zero_shot':
        prompt = f"""You are asked to describe user interests and preferences based on his/her {past_signal} reviews list, 
            Your'e given the user's past {past_signal} reviews in the format: <product category, product title> : <product review content>
            You can only response the user interests and preferences (at most 10 sentences). Don't use lists, use summary structure.
            The output should begin with the word Profile:.
            These are the {past_signal} reviews : \n {history}.
            """
        if reasoning == 'v1':
            prompt = f"{prompt} Let's think step by step"
        elif reasoning == 'v2':
            prompt = f"""{prompt} 
                Let's first understand the problem and devise a plan to solve the problem. 
                Then, let's carry out the plan and solve the problem step by step.
                """
    elif shot == 'one_shot':
        prompt = f"""You are asked to describe user interests and preferences based on his/her {past_signal} reviews list, 
            Your'e given the user's past {past_signal} reviews in the format: <product category, product title> : <product review content>
            You can only response the user interests and preferences (at most 10 sentences). Don't use lists, use summary structure.
            The output should begin with the word Profile:.
            For example, the output profile should be as following:
            The user is a practical and curious individual who values quality, efficiency, and creativity. 
            They enjoy experiences that combine functionality with enjoyment and have a reflective and imaginative approach to life. 
            With an appreciation for innovation and resourcefulness, they are drawn to solutions and narratives that provide value, 
            versatility, and unique perspectives.
            These are the {past_signal} reviews : \n {history}.
            """
    elif shot == 'two_shot':
        prompt = f"""You are asked to describe user interests and preferences based on his/her {past_signal} reviews list, 
            Your'e given the user's past {past_signal} reviews in the format: <product category, product title> : <product review content>
            You can only response the user interests and preferences (at most 10 sentences). Don't use lists, use summary structure.
            The output should begin with the word Profile:.
            For example, the output profile should be as the followings:
            1. The user is a practical and curious individual who values quality, efficiency, and creativity. 
            They enjoy experiences that combine functionality with enjoyment and have a reflective and imaginative approach to life. 
            With an appreciation for innovation and resourcefulness, they are drawn to solutions and narratives that provide value, 
            versatility, and unique perspectives.
            2. The user is a thoughtful and resourceful individual who values meaningful experiences, quality, and practicality. 
            They have a strong appreciation for exploring diverse cultures and destinations, with a knack for finding value and unique perspectives in their choices. 
            Their taste reflects a balance of sophistication and simplicity, enjoying classic elegance and practicality in equal measure. 
            They exhibit a reflective and considerate personality, emphasizing the importance of careful planning and making thoughtful decisions. 
            Creativity, emotional connection, and the ability to find joy in both small details and grand experiences define their preferences.
            These are the {past_signal} reviews : \n {history}.
            """

    try:
        output = llm(f"<s>[INST] {prompt} [/INST]", max_tokens=-1)
    except:
        output = ''

    return output
    
def build_user_profiles(
    llm, category, signal, history_size, his_order, shot, reason, users,
    written_history, voted_history, review_id_text_mapping, output_path
):
    profiles = []
    
    temp_file = os.path.join(output_path, 'profiles_temp.csv')
    if os.path.exists(temp_file):
        temp = pd.read_csv(temp_file)
        profiles = list(temp.itertuples(index=False, name=None))
        
        users = [u for u in users if u not in temp['user_id'].tolist()]
    
    for idx, user in tqdm(enumerate(users)):
        history_info = load_history(user, review_id_text_mapping, history_size, his_order, written_history, 
                                    voted_history, signal, category)
        history = [f'<{h[0]}, {h[1]}>: <{h[2].strip()}>' for h in history_info]
        history = '\n'.join(history)
        
        profile = build_profile(llm, signal, history, shot, reason)
        profiles.append((user, history_info, profile))
        
        if idx % 10 == 0:
           pd.DataFrame(profiles, columns=['user_id', 'history', 'profile']).to_csv(
               os.path.join(output_path, 'profiles_temp.csv'), index=False)
        
    return pd.DataFrame(profiles, columns=['user_id', 'history', 'profile'])

def post_process_profile(profile):
    # Remove Profile: prefix
    profile = re.sub(r'^\s*Profile:\s*', '', profile)
    profile = profile.strip()

    # Remove numbered lists 
    profile = re.sub(r'\n\d+\.', '', profile)

    # Remove bullet lists
    profile = re.sub(r'\n\*\.', '', profile)
    profile = re.sub(r'\*', '', profile)

    profile = profile.replace('\n', ' ').replace('\'', '')

    return profile
    
def run(category, signal, history_size, review_len, his_order, shot, reason, output_path):
    model_path = c.LLAMA_MODEL_PATH
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=10000,
        temperature=0.5,
        max_tokens=200,
        n_gpu_layers=-1,
        n_batch=512
    )
    
    users, written_reviews, voted_reviews, review_id_text_mapping = load_data(review_len)
    return build_user_profiles(
        llm, category, signal, history_size, his_order, shot, reason, users, 
        written_reviews, voted_reviews, review_id_text_mapping, output_path
    )
    
    
if __name__ == '__main__':
    print('Starting profile generation usign Llama CPP')
    ap = argparse.ArgumentParser(description='User profile generation')
    ap.add_argument('--category', default='Beauty', help='Name of Ciao category. Default is Beauty.')
    ap.add_argument('--signal', default='vote', help='The interaction between the user and the review. Default is vote')
    ap.add_argument('--his_size', default=5, type=int, help='The number of reviews used as user history')
    ap.add_argument('--review_len', default=150, type=int, help='Length of review to include in the prompt')
    ap.add_argument('--his_order', default='rel_last', help='Order of history reviews')
    ap.add_argument('--shot', default='one_shot', help='Whether to use examples in prompts or not')
    ap.add_argument('--reason', default=None, help='Whether add a reasoning to the prompt or not')
    
    args = ap.parse_args()
    print(f'DEBUG: args: {args}')
    
    output_path = os.path.join(
        c.PROFILES_PATH, f'ciao_{args.category.replace(" & ", "_")}',
        f'his_size_{args.his_size}_rev_len_{args.review_len}_order_{args.his_order}_{args.shot}_v2'
    )
    if args.reason:
        output_path = f'{output_path}_reason_{args.reason}'
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df = run(
        args.category, args.signal, args.his_size, args.review_len, args.his_order, 
        args.shot, args.reason, output_path
    )
    
    df['profile'] = df['profile'].apply(post_process_profile)
    df = df.drop_duplicates(subset='user_id', keep='first')
    df.to_csv(os.path.join(output_path, f'{args.signal}_profiles.csv'), index=False)
    
    os.remove(os.path.join(output_path, 'profiles_temp.csv'))
    print('done')
