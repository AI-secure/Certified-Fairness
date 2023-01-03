import numpy as np
import random


# TODO: check this file (the way of generating shifted distribution) carefully

def get_label_distribution_Q(lamb_Q, lamb_P, count_P, n_data_P):
    pivot = np.argmax(np.array(lamb_Q) / np.array(lamb_P))
    count_Q = [0] * 4
    count_Q[pivot] = count_P[pivot]  # TODO: count_Q[pivot] = count_P[pivot] or random.randint(0,count_P[pivot])
    for i in range(4):
        count_Q[i] = int((1.0 * lamb_Q[i] / lamb_Q[pivot]) * count_Q[pivot])
    return count_Q

def get_label_distribution_Q_2(lamb_Q, lamb_P, count_P, n_data_P):
    pivot = np.argmax(np.array(lamb_Q) / np.array(lamb_P))
    count_Q = [0] * 2
    count_Q[pivot] = count_P[pivot]  # TODO: count_Q[pivot] = count_P[pivot] or random.randint(0,count_P[pivot])
    for i in range(2):
        count_Q[i] = int((1.0 * lamb_Q[i] / lamb_Q[pivot]) * count_Q[pivot])
    return count_Q

def hellinger_label_shifting(label_P, label_Q):
    label_P, label_Q = np.array(label_P), np.array(label_Q)
    squares = (np.sqrt(label_P) - np.sqrt(label_Q)) ** 2
    dist = np.sqrt(0.5 * np.sum(squares))
    return dist

def label_shifting_binary_label_binary_attr(seed, data_P, group_00, group_01, group_10, group_11, gamma_k, gamma_r):
    random.seed(seed)
    k = random.uniform(gamma_k,1-gamma_k)

    # k = 0.5
    r = random.uniform(gamma_r,1-gamma_r)

    lamb_Q = [k * r, k * (1-r), (1-k) * r, (1-k) * (1-r)]

    # Process distrition data_P
    count_P = [len(group_00), len(group_01), len(group_10), len(group_11)]
    n_data_P = count_P[0] + count_P[1] + count_P[2] + count_P[3]
    lamb_P = [1.0 * count_P[i] / n_data_P for i in range(4)]

    # Get label distribution of data_Q
    count_Q = get_label_distribution_Q(lamb_Q, lamb_P, count_P, n_data_P)
    # print(f'Label distribution of Q: {count_Q}')

    # Calculate Hellinger Distance (label shifting)
    # print(lamb_Q)
    # print(lamb_P)
    # label_Q = [lamb_Q[0]+lamb_Q[2], lamb_Q[1]+lamb_Q[3]]
    # label_P = [lamb_P[0]+lamb_P[2], lamb_P[1]+lamb_P[3]]
    label_Q = list(lamb_Q)
    label_P = list(lamb_P)
    dist = hellinger_label_shifting(label_P, label_Q)
    # print(label_Q)
    # print(label_P)
    # print(f'Hellinger distance (label shifting): {dist}')

    indices_Q = []

    group_00_shuffle = list(group_00)
    random.shuffle(group_00_shuffle)
    indices_Q = indices_Q + group_00_shuffle[:count_Q[0]]

    group_01_shuffle = list(group_01)
    random.shuffle(group_01_shuffle)
    indices_Q = indices_Q + group_01_shuffle[:count_Q[1]]

    group_10_shuffle = list(group_10)
    random.shuffle(group_10_shuffle)
    indices_Q = indices_Q + group_10_shuffle[:count_Q[2]]

    group_11_shuffle = list(group_11)
    random.shuffle(group_11_shuffle)
    indices_Q = indices_Q + group_11_shuffle[:count_Q[3]]

    return indices_Q, dist

def label_shifting_binary_label_binary_attr_unbalanced(seed, data_P, group_00, group_01, group_10, group_11, gamma_k, gamma_r):
    random.seed(seed)

    # k = 0.5
    k = random.uniform(gamma_k,1-gamma_k)

    if random.uniform(0,1)<0.5:
        r = random.uniform(0,gamma_r)
    else:
        r = random.uniform(1-gamma_r,1)

    lamb_Q = [k * r, k * (1-r), (1-k) * r, (1-k) * (1-r)]

    # Process distrition data_P
    count_P = [len(group_00), len(group_01), len(group_10), len(group_11)]
    n_data_P = count_P[0] + count_P[1] + count_P[2] + count_P[3]
    lamb_P = [1.0 * count_P[i] / n_data_P for i in range(4)]

    # Get label distribution of data_Q
    count_Q = get_label_distribution_Q(lamb_Q, lamb_P, count_P, n_data_P)
    # print(f'Label distribution of Q: {count_Q}')

    # Calculate Hellinger Distance (label shifting)
    # print(lamb_Q)
    # print(lamb_P)
    # label_Q = [lamb_Q[0]+lamb_Q[2], lamb_Q[1]+lamb_Q[3]]
    # label_P = [lamb_P[0]+lamb_P[2], lamb_P[1]+lamb_P[3]]
    label_Q = list(lamb_Q)
    label_P = list(lamb_P)
    dist = hellinger_label_shifting(label_P, label_Q)
    # print(label_Q)
    # print(label_P)
    # print(f'Hellinger distance (label shifting): {dist}')

    indices_Q = []

    group_00_shuffle = list(group_00)
    random.shuffle(group_00_shuffle)
    indices_Q = indices_Q + group_00_shuffle[:count_Q[0]]

    group_01_shuffle = list(group_01)
    random.shuffle(group_01_shuffle)
    indices_Q = indices_Q + group_01_shuffle[:count_Q[1]]

    group_10_shuffle = list(group_10)
    random.shuffle(group_10_shuffle)
    indices_Q = indices_Q + group_10_shuffle[:count_Q[2]]

    group_11_shuffle = list(group_11)
    random.shuffle(group_11_shuffle)
    indices_Q = indices_Q + group_11_shuffle[:count_Q[3]]

    return indices_Q, dist













def label_shifting_binary_label_binary_attr_different_base_rates(seed, data_P, group_00, group_01, group_10, group_11):
    random.seed(seed)
    k = random.uniform(0.0,1.0)
    r = random.uniform(0.0,1.0-k)
    t = random.uniform(0.0,1.0-k-r)

    lamb_Q = [k, r, t, 1-k-r-t]

    # Process distrition data_P
    count_P = [len(group_00), len(group_01), len(group_10), len(group_11)]
    n_data_P = count_P[0] + count_P[1] + count_P[2] + count_P[3]
    lamb_P = [1.0 * count_P[i] / n_data_P for i in range(4)]

    # Get label distribution of data_Q
    count_Q = get_label_distribution_Q(lamb_Q, lamb_P, count_P, n_data_P)
    # print(f'Label distribution of Q: {count_Q}')

    # Calculate Hellinger Distance (label shifting)
    # print(lamb_Q)
    # print(lamb_P)
    # label_Q = [lamb_Q[0]+lamb_Q[2], lamb_Q[1]+lamb_Q[3]]
    # label_P = [lamb_P[0]+lamb_P[2], lamb_P[1]+lamb_P[3]]
    label_Q = list(lamb_Q)
    label_P = list(lamb_P)
    dist = hellinger_label_shifting(label_P, label_Q)
    # print(label_Q)
    # print(label_P)
    # print(f'Hellinger distance (label shifting): {dist}')

    indices_Q = []

    for i in range(4):
        if count_Q[i]==0:
            count_Q[i]=1

    group_00_shuffle = list(group_00)
    random.shuffle(group_00_shuffle)
    indices_Q = indices_Q + group_00_shuffle[:count_Q[0]]

    group_01_shuffle = list(group_01)
    random.shuffle(group_01_shuffle)
    indices_Q = indices_Q + group_01_shuffle[:count_Q[1]]

    group_10_shuffle = list(group_10)
    random.shuffle(group_10_shuffle)
    indices_Q = indices_Q + group_10_shuffle[:count_Q[2]]

    group_11_shuffle = list(group_11)
    random.shuffle(group_11_shuffle)
    indices_Q = indices_Q + group_11_shuffle[:count_Q[3]]


    return indices_Q, dist, 1.0*count_Q[1]/(count_Q[0]+count_Q[1]) - 1.0*count_Q[3]/(count_Q[2]+count_Q[3])




def label_shifting_binary_label_binary_attr_class_parity(seed, data_P, group_00, group_01, group_10, group_11):
    random.seed(seed)


    # Process distrition data_P
    count_P = [len(group_00)+len(group_10), len(group_01)+len(group_11)]
    n_data_P = count_P[0] + count_P[1]
    lamb_P = [1.0 * count_P[i] / n_data_P for i in range(2)]

    x = random.uniform(min(lamb_P[0],lamb_P[1]),max(lamb_P[0],lamb_P[1]))
    lamb_Q = [x, 1-x]
    # lamb_Q = lamb_P

    # Get label distribution of data_Q
    count_Q = get_label_distribution_Q_2(lamb_Q, lamb_P, count_P, n_data_P)
    # print(f'Label distribution of Q: {count_Q}')

    # Calculate Hellinger Distance (label shifting)
    # print(lamb_Q)
    # print(lamb_P)
    # label_Q = [lamb_Q[0]+lamb_Q[2], lamb_Q[1]+lamb_Q[3]]
    # label_P = [lamb_P[0]+lamb_P[2], lamb_P[1]+lamb_P[3]]
    label_Q = list(lamb_Q)
    label_P = list(lamb_P)
    dist = hellinger_label_shifting(label_P, label_Q)
    # print(label_Q)
    # print(label_P)
    # print(f'Hellinger distance (label shifting): {dist}')

    indices_Q_0 = []
    indices_Q_1 = []

    group_0_shuffle = list(group_00) + list(group_10)
    random.shuffle(group_0_shuffle)
    indices_Q_0 = indices_Q_0 + group_0_shuffle[:count_Q[0]]

    group_1_shuffle = list(group_01) + list(group_11)
    random.shuffle(group_1_shuffle)
    indices_Q_1 = indices_Q_1 + group_1_shuffle[:count_Q[1]]

    return indices_Q_0, indices_Q_1, dist


# def label_shifting_binary_label_binary_attr_class_parity(seed, data_P, group_00, group_01, group_10, group_11):
#     random.seed(seed)
#     k = random.uniform(0.0,1.0)
#     r = random.uniform(0.0,1.0)
#
#     lamb_Q = [k * r, k * (1-r), (1-k) * r, (1-k) * (1-r)]
#
#     # Process distrition data_P
#     count_P = [len(group_00), len(group_01), len(group_10), len(group_11)]
#     n_data_P = count_P[0] + count_P[1] + count_P[2] + count_P[3]
#     lamb_P = [1.0 * count_P[i] / n_data_P for i in range(4)]
#
#     # Get label distribution of data_Q
#     count_Q = get_label_distribution_Q(lamb_Q, lamb_P, count_P, n_data_P)
#     # print(f'Label distribution of Q: {count_Q}')
#
#     # Calculate Hellinger Distance (label shifting)
#     # print(lamb_Q)
#     # print(lamb_P)
#     # label_Q = [lamb_Q[0]+lamb_Q[2], lamb_Q[1]+lamb_Q[3]]
#     # label_P = [lamb_P[0]+lamb_P[2], lamb_P[1]+lamb_P[3]]
#     label_Q = list(lamb_Q)
#     label_P = list(lamb_P)
#     dist = hellinger_label_shifting(label_P, label_Q)
#     # print(label_Q)
#     # print(label_P)
#     # print(f'Hellinger distance (label shifting): {dist}')
#
#     indices_Q_0 = []
#     indices_Q_1 = []
#
#     group_00_shuffle = list(group_00)
#     random.shuffle(group_00_shuffle)
#     indices_Q_0 = indices_Q_0 + group_00_shuffle[:count_Q[0]]
#
#     group_01_shuffle = list(group_01)
#     random.shuffle(group_01_shuffle)
#     indices_Q_0 = indices_Q_0 + group_01_shuffle[:count_Q[1]]
#
#     group_10_shuffle = list(group_10)
#     random.shuffle(group_10_shuffle)
#     indices_Q_1 = indices_Q_1 + group_10_shuffle[:count_Q[2]]
#
#     group_11_shuffle = list(group_11)
#     random.shuffle(group_11_shuffle)
#     indices_Q_1 = indices_Q_1 + group_11_shuffle[:count_Q[3]]
#
#     return indices_Q_0, indices_Q_1, dist
