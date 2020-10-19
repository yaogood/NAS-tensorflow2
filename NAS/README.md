
| File                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| controller.py        | `Controller` manages the training and evaluation of the Controller RNN |
| dense_env.py         | `DenseEnv` represent the environment of fully-connected network |
| global_parameters.py | Some hyper-parameters for the whole model                    |
| load_data.py         | Loading datasets                                             |
| cnn_env.py           | Containing the environment of fully-connected network        |
| search_space.py      | Defining the Search Space                                    |
| train.py             | Training                                                     |
| utils.py             | Some help functions                                          |


### Usage

For full training details, please see `train.py`.  Bellowing code defines the search space of fully-connected networks

```python
# construct a state space
ss = IdenticalLayerSearchSpace(num_layers=args.num_layers)
ss.add_state(name='neurons', values=range(32, 512, 32))
ss.print_search_space()
```


