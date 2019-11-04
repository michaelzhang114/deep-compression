import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    
    # rounding errors
    float_offset = 0.0001    
    k = 2**bits

    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            pass
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            weights = m.conv.weight.detach().cpu().numpy()
            weights_copy = weights.reshape(-1,1)

            max_weight = np.amax(weights_copy)
            min_weight = np.amin(weights_copy)
            range_weights = max_weight - min_weight
            counter = min_weight
            linear_initialization = []
            while counter <= max_weight + float_offset:
                linear_initialization.append(counter)
                counter += range_weights / (k-1)

            init_np = np.array(linear_initialization).reshape(-1,1)[0:k]
            weights_sparse = [value[0] for value in weights_copy if value != 0.]
            zero_indices = np.where(weights_copy == 0.)[0]
            non_zero_indices = np.where(weights_copy != 0.)[0]

            kmeans = KMeans(n_clusters=k, init=init_np, n_init=1).fit(np.array(weights_sparse).reshape(-1,1))
            centroids = kmeans.cluster_centers_
            cluster_centers.append(centroids)

            cluster_index = kmeans.labels_
            quantized_weights = np.empty(len(weights_sparse))

            for i in range(len(weights_sparse)):
                val = cluster_index[i]
                quantized_weights[i] = (centroids[val][0])

            weights_out = np.zeros(len(weights_copy))

            for i in range(len(non_zero_indices)):    
                loc = non_zero_indices[i]
                weights_out[loc] = quantized_weights[i]

            quantized_weights_torch = torch.from_numpy(weights_out.reshape(weights.shape)).cuda()
            quantized_weights_torch = quantized_weights_torch.float()
            m.conv.weight.data = quantized_weights_torch
            """-----------------------------------------"""
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            weights = m.linear.weight.detach().cpu().numpy()
            weights_copy = weights.reshape(-1,1)

            max_weight = np.amax(weights_copy)
            min_weight = np.amin(weights_copy)
            range_weights = max_weight - min_weight
            counter = min_weight
            linear_initialization = []
            while counter <= max_weight + float_offset:
                linear_initialization.append(counter)
                counter += range_weights / (k-1)

            init_np = np.array(linear_initialization).reshape(-1,1)[0:k]
            weights_sparse = [value[0] for value in weights_copy if value != 0.]
            zero_indices = np.where(weights_copy == 0.)[0]
            non_zero_indices = np.where(weights_copy != 0.)[0]

            kmeans = KMeans(n_clusters=k, init=init_np, n_init=1).fit(np.array(weights_sparse).reshape(-1,1))
            centroids = kmeans.cluster_centers_
            cluster_centers.append(centroids)

            cluster_index = kmeans.labels_
            quantized_weights = np.empty(len(weights_sparse))

            for i in range(len(weights_sparse)):
                val = cluster_index[i]
                quantized_weights[i] = (centroids[val][0])

            weights_out = np.zeros(len(weights_copy))

            for i in range(len(non_zero_indices)):    
                loc = non_zero_indices[i]
                weights_out[loc] = quantized_weights[i]

            quantized_weights_torch = torch.from_numpy(weights_out.reshape(weights.shape)).cuda()
            quantized_weights_torch = quantized_weights_torch.float()
            m.linear.weight.data = quantized_weights_torch
            """----------------------------------------"""
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

