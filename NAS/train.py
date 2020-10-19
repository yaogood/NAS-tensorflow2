import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorboardX import SummaryWriter
from tensorflow.keras import Sequential, layers, optimizers, datasets

from global_parameters import parse_args
from dense_env import DenseEnv
from load_data import load_structure_dataset, load_covertype, load_sensit, load_intrusion
from search_space import IdenticalLayerSearchSpace
from controller import RNNPolicyNetwork, train_controller

print(tf.version.VERSION)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

checkpoint_dir = 'controller_weights/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

args = parse_args()
# construct a state space
ss = IdenticalLayerSearchSpace(num_layers=args.num_layers)
ss.add_state(name='neurons', values=range(32, 512, 32))
ss.print_search_space()


def train(dataset, epochs, total_steps, save_filename=None, load_filename=None, hidden=None, policy_net=None):
    if not policy_net:
        print("\ninitial model weights successfully.\n")
        policy_net = RNNPolicyNetwork(units=args.lstm_units, search_space=ss,
                                      batch_size=args.controller_batch_size, embedding_dim=args.embedding_dim)
    if load_filename:
        print("\nload ", load_filename, " model weights successfully.")
        checkpoint_path = os.path.join(checkpoint_dir, load_filename)
        policy_net.load_weights(checkpoint_path)

    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.03, decay_steps=100,
                                                        decay_rate=0.95, staircase=True)
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    env = DenseEnv(dataset, epochs=epochs, batchsize=args.dataset_batch_size)

    # train the controller on the saved state and the discounted rewards
    arch_buffer, accuracy_buffer, ema_reward_buffer, loss_buffer, hidden \
        = train_controller(policy_net, ss, env, optimizer, steps=total_steps, reg_strength=args.reg_strength,
                           clip_norm=args.clip_norm, entropy_coeff=args.entropy_coeff,
                           ema_alpha=args.ema_alpha, hidden=hidden, more_exploration=args.more_exploration)

    best_archs = policy_net(None, training=False, prev_hidden=hidden)
    best_actions = None
    best_acc = 0.0
    for arch in best_archs:
        actions = ss.to_values(arch.numpy())
        acc = env.get_rewards(actions)
        if acc > best_acc:
            best_acc = acc
            best_actions = actions
    print("++++++++++++++++++++++++++++++++++++++++")
    print(best_actions, " ==> ", best_acc)
    print("++++++++++++++++++++++++++++++++++++++++")

    print(arch_buffer)
    print(accuracy_buffer)
    print(ema_reward_buffer)
    print(loss_buffer)

    # policy_net.predict(1,batch_size=1)

    if save_filename:
        print("\nsave ", save_filename, " model weights successfully.")
        checkpoint_path = os.path.join(checkpoint_dir, save_filename)
        policy_net.save_weights(checkpoint_path)

    return hidden, policy_net


if __name__ == '__main__':
    ############################ covertype ##################################
    # x1, y1, x2, y2, x_test, y_test = load_sensit()
    #
    # def sf(x, num_features): # select features
    #     # choose features
    #     if num_features == 12:
    #         selected_idx = np.arange(54)
    #     elif num_features == 11:
    #         selected_idx = np.arange(14)
    #     else:
    #         selected_idx = np.arange(num_features)
    #     return x[:, selected_idx]
    #
    # datasets = (sf(x1, 6), y1, sf(x_test, 6), y_test)
    # hidden, net = train(datasets, epochs=8, total_steps=100, save_filename='coverType_6.hdf5')
    #
    # datasets = (x2, y2, x_test, y_test) # [60, 45, 45, 61]
    # _, _ = train(datasets, epochs=8, total_steps=100, load_filename='coverType_6.hdf5', hidden=hidden, policy_net=net)
    #
    # datasets = (sf(x1, 8), y1, sf(x_test, 8), y_test)
    # hidden, net = train(datasets, epochs=8, total_steps=100, save_filename='coverType_8.hdf5')
    #
    # datasets = (x2, y2, x_test, y_test) # [29, 45, 53, 53]
    # _, _ = train(datasets, epochs=8, total_steps=100, load_filename='coverType_8.hdf5', hidden=hidden, policy_net=net)
    #
    # datasets = (sf(x1, 10), y1, sf(x_test, 10), y_test)
    # hidden, net = train(datasets, epochs=8, total_steps=100, save_filename='coverType_10.hdf5')
    #
    # datasets = (x2, y2, x_test, y_test) # [27, 45, 53, 57]
    # _, _ = train(datasets, epochs=8, total_steps=100, load_filename='coverType_10.hdf5', hidden=hidden, policy_net=net)
    #
    # datasets = (x2, y2, x_test, y_test) #
    # _, _ = train(datasets, epochs=8, total_steps=100, save_filename='coverType.hdf5')

    ################################ sensIT ################################
    # x1, y1, x2, y2, x_test, y_test = load_sensit()
    #
    # def sf(x, num_features):  # select features
    #     selected_idx = np.arange(num_features)
    #     return x[:, selected_idx]
    #
    # datasets = (sf(x1, 55), y1, sf(x_test, 55), y_test)
    # hidden, net = train(datasets, epochs=10, total_steps=100, save_filename='sensIT_55.hdf5')
    #
    # datasets = (x2, y2, x_test, y_test)  # [60, 45, 45, 61]
    # _, _ = train(datasets, epochs=10, total_steps=100, load_filename='sensIT_55.hdf5', hidden=hidden, policy_net=net)
    #
    # datasets = (sf(x1, 70), y1, sf(x_test, 70), y_test)
    # hidden, net = train(datasets, epochs=10, total_steps=100, save_filename='sensIT_70.hdf5')
    #
    # datasets = (x2, y2, x_test, y_test)  # [29, 45, 53, 53]
    # _, _ = train(datasets, epochs=10, total_steps=100, load_filename='sensIT_70.hdf5', hidden=hidden, policy_net=net)
    #
    # datasets = (sf(x1, 85), y1, sf(x_test, 85), y_test)
    # hidden, net = train(datasets, epochs=10, total_steps=100, save_filename='sensIT_85.hdf5')
    #
    # datasets = (x2, y2, x_test, y_test)  # [27, 45, 53, 57]
    # _, _ = train(datasets, epochs=10, total_steps=100, load_filename='sensIT_85.hdf5', hidden=hidden, policy_net=net)
    #
    # datasets = (x2, y2, x_test, y_test)  #
    # _, _ = train(datasets, epochs=10, total_steps=100, save_filename='sensIT.hdf5')

    while (1):
        x1, y1, x2, y2, x_test, y_test = load_intrusion()
        dataset = (x1, y1, x_test, y_test)
        env = DenseEnv(dataset, epochs=5, batchsize=128)
        r1 = env.get_rewards([256, 256, 256, 256])
        dataset = (x2, y2, x_test, y_test)
        env = DenseEnv(dataset, epochs=5, batchsize=128)
        r2 = env.get_rewards([256, 256, 256, 256])

        if abs(r1 - r2) < 0.02:
            break


    def sf(x, num_features):  # select features
        selected_idx = np.arange(num_features)
        return x[:, selected_idx]


    datasets = (sf(x1, 60), y1, sf(x_test, 60), y_test) # 128, 160, 96, 448
    hidden, net = train(datasets, epochs=8, total_steps=100, save_filename='netIntru_60.hdf5')

    datasets = (x2, y2, x_test, y_test)
    _, _ = train(datasets, epochs=8, total_steps=100, load_filename='netIntru_60.hdf5', hidden=hidden, policy_net=net)

    datasets = (sf(x1, 80), y1, sf(x_test, 80), y_test) # 384, 96, 160, 96
    hidden, net = train(datasets, epochs=8, total_steps=100, save_filename='netIntru_80.hdf5')

    datasets = (x2, y2, x_test, y_test)
    _, _ = train(datasets, epochs=8, total_steps=100, load_filename='netIntru_80.hdf5', hidden=hidden, policy_net=net)

    datasets = (sf(x1, 100), y1, sf(x_test, 100), y_test) # 64, 448, 192, 352
    hidden, net = train(datasets, epochs=8, total_steps=100, save_filename='netIntru_100.hdf5')

    datasets = (x2, y2, x_test, y_test)
    _, _ = train(datasets, epochs=8, total_steps=100, load_filename='netIntru_100.hdf5', hidden=hidden, policy_net=net)

    datasets = (x2, y2, x_test, y_test)  # 320, 384, 32, 288
    _, _ = train(datasets, epochs=8, total_steps=100, save_filename='netIntru.hdf5')