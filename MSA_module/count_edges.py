import numpy as np
a = np.load('train_data.npz', allow_pickle=True)
b = np.load('train_edge_lists.npz', allow_pickle=True)
for i in range(len(b['edges'])):
    count = sum(np.sum(data > -1) for data in a['ss_data'][i])
    largest = b['edges'][i][-1][0]
    if count + 1 < largest:
        print(f'There are problem in {i} th chain, the name of ss is {a['identifiers'][i]}, the name of edge is {b['identifiers'][i]}, count = {count}, largest = {largest}')
