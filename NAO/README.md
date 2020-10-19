
| File            | Description                              |
| --------------- | ---------------------------------------- |
| decoder.py      | Decoder of the auto-encoder              |
| encoder.py      | Encoder of the auto-encoder              |
| load_data.py    | Loading datasets                         |
| model.py        | CNN modules                              |
| model_search.py | Shared CNN modules                       |
| nao.py          | NAO models                               |
| operations.py   | Basic operations implementation          |
| search.py       | Searching for architectures              |
| search_space.py | Search Space                             |
| train.py        | Trainng the NAO models                   |
| train_cifar.py  | Training cifar-10 and cifar-100 datasets |
| train_cifar.sh  | shell file to run                        |
| train_nao.py    | Trainng the NAO models                   |
| utils.py        | Help functions                           |

### Usage

For full training details, please see `train_cifar.py`. 

```shell
bash train_cifar.sh
```


Re-implementation of [renqianluo](https://github.com/renqianluo)/**[NAO_pytorch](https://github.com/renqianluo/NAO_pytorch)**

The performance is lower than the initial work. There are still some bugs for these codes. But the basic modules and classes can be reused in other works. Highly recommend using the pytorch version work from [renqianluo](https://github.com/renqianluo)/**[NAO_pytorch](https://github.com/renqianluo/NAO_pytorch)**.
