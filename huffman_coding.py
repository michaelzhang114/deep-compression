import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn
# for min heap 
import heapq

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class _huffman_node:
    # priority(freq) -> item(weight)
    def __init__(self, item, freq, left=None, right=None):
        self.item = item
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq
    
def _make_heap(weights_frequency):
    heap = []
    for keyy in weights_frequency:
        heapq.heappush(heap, _huffman_node(item=keyy, freq=weights_frequency[keyy]))
    return heap
    
def _merge_heap_nodes(heap):
    while len(heap) > 1:
        min_node_1 = heapq.heappop(heap)
        min_node_2 = heapq.heappop(heap)
        combined_freq = min_node_1.freq + min_node_2.freq        
        combined_node = _huffman_node(item=None, freq=combined_freq, left=min_node_1, right=min_node_2)
        heapq.heappush(heap, combined_node) 
    return heap

def _walk_tree(node, code, encodings):
    if node is None:
        return
    
    _walk_tree(node.left, code + '0', encodings)
    _walk_tree(node.right, code + '1', encodings)
    
    if node.item is not None:
        encodings[node.item] = code

def _get_encodings(node):
    encodings = {}
    _walk_tree(node, '', encodings)
    return encodings

def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: 
            'encodings': Encoding map mapping each weight parameter to its Huffman coding.
            'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    """
    """
    Generate Huffman Coding and Frequency Map according to incoming weights and centers (KMeans centriods).
    --------------Your Code---------------------
    """
    
    # do I quantize the weights first??
    #weights = np.array([2,3,4,4,4,4,4,4,1,2,3,7,6,6,6,6,3,5,5,5,5,5,5,5,5,5])   
    weights = weight
    elements, counts = np.unique(weights, return_counts=True)
    
    # frequency
    frequency = {}
    i = 0
    for item in elements:
        frequency[item] = counts[i]
        i += 1
    
    # encodings
    heap = _make_heap(frequency)
    final_heap = _merge_heap_nodes(heap)
    n = heapq.heappop(final_heap)
    encodings = _get_encodings(n)
    
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param 'encodings': Encoding map mapping each weight parameter to its Huffman coding.
    :param 'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    orig_storage = []
    new_storage = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            orig_storage.append(orginal_avg_bits)
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            #print(encodings)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            new_storage.append(huffman_avg_bits)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            orig_storage.append(orginal_avg_bits)
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            #print(encodings)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            new_storage.append(huffman_avg_bits)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
    
    old_bits = sum(orig_storage) / len(orig_storage)
    new_bits = sum(new_storage) / len(new_storage)
    return freq_map, encodings_map, old_bits, new_bits