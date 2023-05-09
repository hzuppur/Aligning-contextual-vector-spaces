import numpy as np
import statistics


def relative_euclidean_distance(vec_1_lst, vec_2_lst):
    zero_vector = np.zeros(vec_1_lst.shape[1])
    euclidean_distance_sum = 0

    for a, b in zip(vec_1_lst, vec_2_lst):
        euclidean_distance_sum += np.linalg.norm(a - b) / np.linalg.norm(a - zero_vector)

    return euclidean_distance_sum / vec_1_lst.shape[0]


def relative_euclidean_distance_stdev(vec_1_lst, vec_2_lst):
    zero_vector = np.zeros(vec_1_lst.shape[1])
    euclidean_distance_list = []

    for a, b in zip(vec_1_lst, vec_2_lst):
        euclidean_distance_list.append(np.linalg.norm(a - b) / np.linalg.norm(a - zero_vector))

    stdev = statistics.stdev(euclidean_distance_list)

    return sum(euclidean_distance_list) / len(euclidean_distance_list), stdev
