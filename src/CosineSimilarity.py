from numpy import dot
from numpy.linalg import norm
import statistics


def cosine_similarity(vec_1_lst, vec_2_lst):
    if len(vec_1_lst) != len(vec_2_lst):
        raise RuntimeError("The two list of vectors must have the same length!")

    sum_cos_sim = 0
    n_vec = len(vec_1_lst)
    for i in range(n_vec):
        a = vec_1_lst[i]
        b = vec_2_lst[i]

        cos_sim = dot(a, b) / (norm(a) * norm(b))
        sum_cos_sim += cos_sim

    return sum_cos_sim / n_vec

def cosine_similarity_with_stdev(vec_1_lst, vec_2_lst):
    cos_sim_list = []
    
    for a, b in zip(vec_1_lst, vec_2_lst):
        cos_sim = dot(a, b) / (norm(a) * norm(b))
        cos_sim_list.append(cos_sim)
        
    stdev = statistics.stdev(cos_sim_list)
    
    return sum(cos_sim_list) / len(cos_sim_list), stdev