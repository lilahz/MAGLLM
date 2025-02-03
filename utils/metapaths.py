import numpy as np
import pathlib

from tqdm import tqdm


# 0-1-0
def create_u_r_u(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create u-r-u list')
    u_r_u = []
    for r, u_list in tqdm(review_user_list.items()):
        u_r_u.extend([(u1, r, u2) for u1 in u_list for u2 in u_list])
    u_r_u = np.array(u_r_u)
    u_r_u[:, 1] += num_users
    sorted_index = sorted(list(range(len(u_r_u))), key=lambda i : u_r_u[i, [0, 2, 1]].tolist())
    u_r_u = u_r_u[sorted_index]
    return u_r_u

# 1-2-1
def create_r_p_r(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create r-p-r list')
    r_p_r = []
    for p, r_list in tqdm(product_review_list.items()):
        r_p_r.extend([(r1, p, r2) for r1 in r_list for r2 in r_list])
    r_p_r = np.array(r_p_r)
    r_p_r += num_users
    r_p_r[:, 1] += num_reviews
    sorted_index = sorted(list(range(len(r_p_r))), key=lambda i : r_p_r[i, [0, 2, 1]].tolist())
    r_p_r = r_p_r[sorted_index]
    return r_p_r

#0-1-2-1-0
def create_u_r_p_r_u(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create u-r-p-r-u list')
    u_r_p_r_u = []
    r_p_r = create_r_p_r(user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews)
    for r1, p, r2 in tqdm(r_p_r):
        if len(review_user_list[r1 - num_users]) == 0 or len(review_user_list[r2 - num_users]) == 0:
            continue
        candidate_u1_list = np.random.choice(len(review_user_list[r1 - num_users]), int(0.2 * len(review_user_list[r1 - num_users])), replace=False)
        candidate_u1_list = review_user_list[r1 - num_users][candidate_u1_list]
        candidate_u2_list = np.random.choice(len(review_user_list[r2 - num_users]), int(0.2 * len(review_user_list[r2 - num_users])), replace=False)
        candidate_u2_list = review_user_list[r2 - num_users][candidate_u2_list]
        u_r_p_r_u.extend([(u1, r1, p, r2, u2) for u1 in candidate_u1_list for u2 in candidate_u2_list])
    u_r_p_r_u = np.array(u_r_p_r_u)
    sorted_index = sorted(list(range(len(u_r_p_r_u))), key=lambda i : u_r_p_r_u[i, [0, 4, 1, 2, 3]].tolist())
    u_r_p_r_u = u_r_p_r_u[sorted_index]
    return u_r_p_r_u

def create_a_r_p_r_a(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create a-r-p-r-a list')
    u_r_p_r_u = []
    r_p_r = create_r_p_r(user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews)
    for r1, p, r2 in r_p_r:
        if len(review_author_list[r1 - num_users]) == 0 or len(review_author_list[r2 - num_users]) == 0:
            continue
        candidate_u1_list = review_author_list[r1 - num_users]
        candidate_u2_list = review_author_list[r2 - num_users]
        u_r_p_r_u.extend([(u1, r1, p, r2, u2) for u1 in candidate_u1_list for u2 in candidate_u2_list])
    u_r_p_r_u = np.array(u_r_p_r_u)
    sorted_index = sorted(list(range(len(u_r_p_r_u))), key=lambda i : u_r_p_r_u[i, [0, 4, 1, 2, 3]].tolist())
    u_r_p_r_u = u_r_p_r_u[sorted_index]
    return u_r_p_r_u

# 1-0-1
def create_r_u_r(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create r-u-r list')
    r_u_r = []
    for u, r_list in tqdm(user_review_list.items()):
        r_u_r.extend([(r1, u, r2) for r1 in r_list for r2 in r_list])
    r_u_r = np.array(r_u_r)
    r_u_r[:, [0, 2]] += num_users
    sorted_index = sorted(list(range(len(r_u_r))), key=lambda i : r_u_r[i, [0, 2, 1]].tolist())
    r_u_r = r_u_r[sorted_index]
    return r_u_r

def create_r_a_r(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create r-a-r list')
    r_u_r = []
    for u, r_list in author_review_list.items():
        r_u_r.extend([(r1, u, r2) for r1 in r_list for r2 in r_list])
    r_u_r = np.array(r_u_r)
    r_u_r[:, [0, 2]] += num_users
    # sorted_index = sorted(list(range(len(r_u_r))), key=lambda i : r_u_r[i, [0, 2, 1]].tolist())
    # r_u_r = r_u_r[sorted_index]
    return r_u_r

# 0-2-0
def create_u_p_u(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create u-p-u list')
    u_p_u = []
    for p, u_list in tqdm(product_user_list.items()):
        u_p_u.extend([(u1, p, u2) for u1 in u_list for u2 in u_list])
    u_p_u = np.array(u_p_u)
    u_p_u[:, [1]] += num_users + num_reviews
    sorted_index = sorted(list(range(len(u_p_u))), key=lambda i : u_p_u[i, [0, 2, 1]].tolist())
    u_p_u = u_p_u[sorted_index]
    return u_p_u

def create_a_p_a(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create a-p-a list')
    u_p_u = []
    for p, u_list in tqdm(product_author_list.items()):
        u_p_u.extend([(u1, p, u2) for u1 in u_list for u2 in u_list])
    u_p_u = np.array(u_p_u)
    u_p_u[:, [1]] += num_users + num_reviews
    sorted_index = sorted(list(range(len(u_p_u))), key=lambda i : u_p_u[i, [0, 2, 1]].tolist())
    u_p_u = u_p_u[sorted_index]
    return u_p_u

# 2-0-2
def create_p_u_p(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create p-u-p list')
    p_u_p = []
    for u, p_list in tqdm(user_product_list.items()):
        p_u_p.extend([(p1, u, p2) for p1 in p_list for p2 in p_list])
    p_u_p = np.array(p_u_p)
    p_u_p[:, [0, 2]] += num_users + num_reviews
    sorted_index = sorted(list(range(len(p_u_p))), key=lambda i : p_u_p[i, [0, 2, 1]].tolist())
    p_u_p = p_u_p[sorted_index]
    return p_u_p

def create_p_a_p(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create p-a-p list')
    p_u_p = []
    for u, p_list in tqdm(author_product_list.items()):
        p_u_p.extend([(p1, u, p2) for p1 in p_list for p2 in p_list])
    p_u_p = np.array(p_u_p)
    p_u_p[:, [0, 2]] += num_users + num_reviews
    sorted_index = sorted(list(range(len(p_u_p))), key=lambda i : p_u_p[i, [0, 2, 1]].tolist())
    p_u_p = p_u_p[sorted_index]
    return p_u_p

# # 1-2-0-2-1
def create_r_p_u_p_r(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create r-p-u-p-r list')
    offset = num_users+num_reviews
    r_p_u_p_r = []
    p_u_p = create_p_u_p(user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews)
    for p1, u, p2 in tqdm(p_u_p):
        if len(product_user_list[p1 - offset]) == 0 or len(product_user_list[p2 - offset]) == 0:
            continue
        candidate_r1_list = np.random.choice(len(product_review_list[p1 - offset]), int(0.2 * len(product_review_list[p1 - offset])), replace=False)
        candidate_r1_list = product_review_list[p1 - offset][candidate_r1_list]
        candidate_r2_list = np.random.choice(len(product_review_list[p2 - offset]), int(0.2 * len(product_review_list[p2 - offset])), replace=False)
        candidate_r2_list = product_review_list[p2 - offset][candidate_r2_list]
        r_p_u_p_r.extend([(r1, p1, u, p2, r2) for r1 in candidate_r1_list for r2 in candidate_r2_list])
    r_p_u_p_r = np.array(r_p_u_p_r)
    sorted_index = sorted(list(range(len(r_p_u_p_r))), key=lambda i : r_p_u_p_r[i, [0, 4, 1, 2, 3]].tolist())
    r_p_u_p_r = r_p_u_p_r[sorted_index]
    return r_p_u_p_r

def create_r_p_a_p_r(
        user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews
    ):
    print('Create r-p-a-p-r list')
    offset = num_users+num_reviews
    r_p_u_p_r = []
    p_u_p = create_p_a_p(user_review_list, review_user_list, author_review_list, review_author_list, user_product_list, 
        product_user_list, product_review_list, author_product_list, product_author_list, 
        num_users, num_reviews)
    for p1, u, p2 in tqdm(p_u_p):
        if len(product_user_list[p1 - offset]) == 0 or len(product_user_list[p2 - offset]) == 0:
            continue
        candidate_r1_list = np.random.choice(len(product_review_list[p1 - offset]), int(0.2 * len(product_review_list[p1 - offset])), replace=False)
        candidate_r1_list = product_review_list[p1 - offset][candidate_r1_list]
        candidate_r2_list = np.random.choice(len(product_review_list[p2 - offset]), int(0.2 * len(product_review_list[p2 - offset])), replace=False)
        candidate_r2_list = product_review_list[p2 - offset][candidate_r2_list]
        r_p_u_p_r.extend([(r1, p1, u, p2, r2) for r1 in candidate_r1_list for r2 in candidate_r2_list])
    r_p_u_p_r = np.array(r_p_u_p_r)
    sorted_index = sorted(list(range(len(r_p_u_p_r))), key=lambda i : r_p_u_p_r[i, [0, 4, 1, 2, 3]].tolist())
    r_p_u_p_r = r_p_u_p_r[sorted_index]
    return r_p_u_p_r

def get_metapaths(signal, output_path):
    if signal == 'vote':
        expected_metapaths = [
            [(0, 1, 0), (0, 1, 2, 1, 0), (0, 2, 0)],
            [(1, 0, 1), (1, 2, 1), (1, 2, 0, 2, 1)]
        ]
        # create the directories if they do not exist
        for i in range(len(expected_metapaths)):
            pathlib.Path(output_path + '/' + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

        metapath_indices_mapping = {(0, 1, 0): create_u_r_u,
                                    (0, 1, 2, 1, 0): create_u_r_p_r_u,
                                    (0, 2, 0): create_u_p_u, 
                                    (1, 0, 1): create_r_u_r,
                                    (1, 2, 1): create_r_p_r,
                                    (1, 2, 0, 2, 1): create_r_p_u_p_r}
    elif signal == 'write':
        expected_metapaths = [
            [(0, 1, 2, 1, 0), (0, 2, 0)],
            [(1, 0, 1), (1, 2, 1), (1, 2, 0, 2, 1)]
        ]
        # create the directories if they do not exist
        for i in range(len(expected_metapaths)):
            pathlib.Path(output_path + '/' + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

        metapath_indices_mapping = {(0, 1, 2, 1, 0): create_a_r_p_r_a,
                                    (0, 2, 0): create_u_p_u, 
                                    (1, 2, 1): create_r_p_r,
                                    (1, 0, 1): create_r_a_r,
                                    (1, 2, 0, 2, 1): create_r_p_u_p_r}
        
    elif signal == 'both':
        expected_metapaths = [
            [(0, 1, 0), (0, 1, 2, 1, 0), (3, 1, 2, 1, 3), (0, 2, 0), (3, 2, 3)],
            [(1, 0, 1), (1, 3, 1), (1, 2, 1), (1, 2, 0, 2, 1), (1, 2, 3, 2, 1)]
        ]
        # create the directories if they do not exist
        for i in range(len(expected_metapaths)):
            pathlib.Path(output_path + '/' + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
        
        metapath_indices_mapping = {(0, 1, 0): create_u_r_u,
                                    (0, 1, 2, 1, 0): create_u_r_p_r_u,
                                    (3, 1, 2, 1, 3): create_a_r_p_r_a,
                                    (0, 2, 0): create_u_p_u, 
                                    (3, 2, 3): create_a_p_a,
                                    (1, 0, 1): create_r_u_r,
                                    (1, 3, 1): create_r_a_r,
                                    (1, 2, 1): create_r_p_r,
                                    (1, 2, 0, 2, 1): create_r_p_u_p_r,
                                    (1, 2, 3, 2, 1): create_r_p_a_p_r}
        
    return expected_metapaths, metapath_indices_mapping