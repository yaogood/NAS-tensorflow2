import numpy as np
from collections import OrderedDict

class IdenticalLayerSearchSpace():
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.states = OrderedDict()
        self.count = 0

    def add_state(self, name, values):
        values_dict = {}
        for i, val in enumerate(values):
            values_dict[i] = val
        metadata = {
            'name': name,
            'size': len(values),
            'values': values_dict,
        }
        self.states[self.count] = metadata
        self.count = self.count + 1

    def get_value(self, global_id, id):
        return self[global_id]['values'][id]

    def get_random_actions(self):
        actions = []
        for id in range(self.count * self.num_layers):
            sample = np.random.choice(self[id]['size'], size=1)
            actions.append(sample[0])
        return actions

    def to_values(self, actions):
        values = []
        for global_id, id in enumerate(actions):
            value = self.get_value(global_id, id)
            values.append(value)
        return values

    def print_search_space(self):
        print('-' * 20, ' state space ', '-' * 20)
        for id, state in self.states.items():
            print(state)
        print('-' * 20, ' end of state space ', '-' * 20)
        print('\n')

    def __getitem__(self, id):
        return self.states[id % self.count]

# search_sapce = IdenticalLayerSearchSpace(num_layers=4)
# search_sapce.add_state(name='node', values=[16,32,48,64,80,96,128])
# search_sapce.add_state(name='filter_size', values=[1,3,5,7,9])
# search_sapce.print_search_space()