from random import random

import numpy as np


def aug_dataset(dataset, kernel_filter, rand_multiplier):
    dataset_np = np.array(dataset)

    for idx, x in enumerate(dataset):

        aug_multiplier = kernel_filter.shape[0]
        k_filter = kernel_filter.flatten()

        input_tensor_all = x[0]
        print(f'Augmenting {idx} out of {dataset_np.shape[0]}')
        output_tensor_all = np.zeros(shape=(3, 32 * aug_multiplier, 32 * aug_multiplier))

        for chnl in range(3):
            input_tensor = input_tensor_all[chnl]
            output_tensor = np.zeros(shape=(32 * aug_multiplier, 32 * aug_multiplier))

            for i in range(32):
                for j in range(32):
                    w = input_tensor[i][j]
                    a = random.random() * rand_multiplier
                    b = random.random() * rand_multiplier
                    c = random.random() * rand_multiplier
                    d = w - a - b - c

                    m = a / k_filter[0]
                    n = b / k_filter[1]
                    o = c / k_filter[2]
                    p = d / k_filter[3]

                    out_i = i * 2
                    out_j = j * 2

                    output_tensor[out_i][out_j] = m
                    output_tensor[out_i][out_j + 1] = n
                    output_tensor[out_i + 1][out_j] = o
                    output_tensor[out_i + 1][out_j + 1] = p

            output_tensor_all[chnl] = output_tensor

        dataset_np[idx][0] = output_tensor_all
    return dataset_np
