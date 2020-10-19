import copy
import numpy as np
import tensorflow as tf
from collections import OrderedDict


class SearchSpace():
    def __init__(self):  # directed acyclic graph
        self.graph = OrderedDict()
        self.num_nodes = 0

        self.features = {}
        self.num_features = 0
        self.vocal_size = 0

    def add_features(self, name, values):
        values_dict = {}
        for i, val in enumerate(values):
            values_dict[i] = val
        self.features[name] = values_dict
        self.num_features = self.num_features + 1
        self.vocal_size = self.vocal_size + len(values)

    def set_graph(self, nodes_list):  # nodes list is a list of feature names
        for i, name in enumerate(nodes_list):
            values_dict = self.features[name]
            metadata = {
                'name': name,
                'len': len(values_dict),
                'values': values_dict,
            }
            self.graph[i] = metadata
            self.num_nodes = self.num_nodes + 1

    def get_value(self, graph_idx, feature_idx):
        return self.graph[graph_idx]['values'][feature_idx]

    def generate_random_arch(self):
        res = []
        for i in range(self.num_nodes):
            len = self.graph[i]['len']
            sample = np.random.choice(len, size=1)
            res.append(sample[0])
        return res

    def idx_to_values(self, graph):
        res = []
        for graph_idx, features_idx in enumerate(graph):
            value = self.get_value(graph_idx, features_idx)
            res.append(value)
        return res

    def print_search_space(self):
        print('-' * 20, ' search space ', '-' * 20)
        print('num of nodes:', self.num_nodes)
        print('vocal size:', self.vocal_size)
        print('-' * 10, ' graph ', '-' * 10)
        for idx, node in self.graph.items():
            print(node)
        print('-' * 10, ' features ', '-' * 10)
        print(self.features)
        print('-' * 20, ' end of search space ', '-' * 20)
        print('\n')


def generate_nao_dataset(self, search_space, total_samples, batch_size=1, train=True, sos_id=0, eos_id=0, swap=False):
    size = search_space.num_nodes
    encoder_input = np.arange(total_samples * size).reshape(total_samples, size)
    encoder_target = np.zeros((total_samples, 1))
    decoder_input = np.arange(total_samples * size).reshape(total_samples, size)
    decoder_target = np.arange(total_samples * size).reshape(total_samples, size)

    for i in range(total_samples):
        arch = search_space.generate_random_arch()
        if self.swap:
            a = np.random.randint(0, 5)
            b = np.random.randint(0, 5)
            arch = arch[:4 * a] + arch[4 * a + 2:4 * a + 4] + \
                   arch[4 * a:4 * a + 2] + arch[4 * (a + 1):20 + 4 * b] + \
                   arch[20 + 4 * b + 2:20 + 4 * b + 4] + arch[20 + 4 * b:20 + 4 * b + 2] + \
                   arch[20 + 4 * (b + 1):]
        encoder_input[i] = arch
        encoder_target[i] = 0.5 #get_acc(arch)
        if train:
            decoder_input[i] = [self.sos_id] + arch[:-1]
        decoder_target[i] = arch

    if train:
        dataset_loader = tf.data.Dataset.from_tensor_slices((encoder_input,encoder_target,decoder_input,decoder_target))
        dataset_loader = dataset_loader.batch(batch_size)
    else:
        dataset_loader = tf.data.Dataset.from_tensor_slices((encoder_input, decoder_target))
        dataset_loader = dataset_loader.batch(batch_size)

    return dataset_loader


search_sapce = SearchSpace()
search_sapce.add_features(name='node', values=[32,48,64,80,96,128,160,192,224,256,320,384,448,512])
# search_sapce.add_features(name='filter_size', values=[3,5])
search_sapce.set_graph(['node', 'filter_size', 'node', 'node', 'filter_size','filter_size'])
search_sapce.print_search_space()
print(search_sapce.idx_to_values(search_sapce.generate_random_arch()))