import numpy as np
import os
import csv
import tensorflow as tf
from keras import backend as K
from data import load_covertype, load_sensit, load_intrusion
from control import Controller, StateSpace
from environment import NetworkManager
import argparse

print(tf.version.VERSION)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 4  # number of layers of the state space
MAX_TRIALS = 100  # maximum number of models generated
MAX_EPOCHS = 8   # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.9  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 64  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.85  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range

# construct a state space
state_space = StateSpace()
# add states
state_space.add_state(name='neurons', values=range(32, 512, 32))
state_space.print_state_space()


def train(dataset1, dataset2, initial_state, if_restore):
    total_reward = 0.0
    with policy_sess.as_default():
        # create the Controller and build the internal policy network
        controller = Controller(policy_sess, NUM_LAYERS, state_space,
                                reg_param=REGULARIZATION,
                                exploration=EXPLORATION,
                                controller_cells=CONTROLLER_CELLS,
                                embedding_dim=EMBEDDING_DIM,
                                restore_controller=if_restore)
    # clear the previous files
    controller.remove_files()
    # create the Network Manager
    manager1 = NetworkManager(dataset1, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                             acc_beta=ACCURACY_BETA)
    manager2 = NetworkManager(dataset2, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                             acc_beta=ACCURACY_BETA)

    result_reward = []
    result_total_reward = []
    result_acc = []
    result_moving_acc = []
    result_explore_acc = []
    result_exploit_acc = []

    flag = None
    manager = None
    for trial in range(MAX_TRIALS):
        print("\nTrial %d:" % (trial + 1))
        if 2*trial < MAX_TRIALS:
            manager = manager1
            if trial % 2 == 0:
                actions = state_space.get_local_state_space_add(int(trial/2), initial_state)
            else:
                actions = state_space.get_local_state_space(int(trial/2), initial_state)
        else:
            manager = manager2
            with policy_sess.as_default():
                K.set_session(policy_sess)
                flag, actions = controller.get_action(state)  # get an action for the previous state

        # print the action probabilities
        # state_space.print_actions(actions)
        print("Actions : ", state_space.parse_state_space_list(actions))
        # build a model, train and get reward and accuracy from the network manager
        reward, previous_acc, moving_acc = manager.get_rewards(state_space.parse_state_space_list(actions))
        print("Rewards : ", reward)
        print("Accuracy : ", previous_acc)
        print("Movingacc :", moving_acc)

        with policy_sess.as_default():
            K.set_session(policy_sess)

            total_reward += reward
            print("Total reward : ", total_reward)

            # actions and states are equivalent, save the state and reward
            state = actions
            controller.store_rollout(state, reward)

            # train the controller on the saved state and the discounted rewards
            loss = controller.train_step()
            print("Controller loss : %0.6f" % (loss))

            # write the results of this trial into a file
            with open('train_history.csv', mode='a+') as f:
                data = [previous_acc, reward]
                data.extend(state_space.parse_state_space_list(state))
                writer = csv.writer(f)
                writer.writerow(data)
        print()
        result_reward.append(reward)
        result_total_reward.append(total_reward)
        result_acc.append(previous_acc)
        result_moving_acc.append(moving_acc)
        if 2*trial >= MAX_TRIALS:
            if not flag:
                result_explore_acc.append(previous_acc)
            else:
                result_exploit_acc.append(previous_acc)

    print("Rewards : ", result_reward)
    print("Total Rewards :",result_total_reward)
    print("Accuracy : ", result_acc)
    print("Moving acc : ", result_moving_acc)
    print("Explore acc :", result_explore_acc)
    print("Exploit acc : ", result_exploit_acc)


if __name__ == '__main__':
    ##################### covertype ################################
    # x1, y1, x2, y2, x_test, y_test = load_covertype()
    # def sf(x, num_features):  # select features
    #     # choose features
    #     if num_features == 12:
    #         selected_idx = np.arange(54)
    #     elif num_features == 11:
    #         selected_idx = np.arange(14)
    #     else:
    #         selected_idx = np.arange(num_features)
    #     return x[:, selected_idx]
    #
    # dataset2 = (x2, y2, x_test, y_test)
    #
    # # dataset1 = (sf(x1, 6), y1, sf(x_test, 6), y_test)
    # # train(dataset1, dataset2, [480, 360, 360, 488], if_restore=False)
    #
    # # dataset1 = (sf(x1, 8), y1, sf(x_test, 8), y_test)
    # # train(dataset1, dataset2, [232, 360, 424, 424], if_restore=False)
    #
    # dataset1 = (sf(x1, 10), y1, sf(x_test, 10), y_test) # [27, 45, 53, 57]
    # train(dataset1, dataset2, [216, 360, 424, 456], if_restore=False)

    ###################### sensIT ##################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=int, default=1)
    parser.add_argument('--l1', type=int, default=1)
    parser.add_argument('--l2', type=int, default=1)
    parser.add_argument('--l3', type=int, default=1)
    parser.add_argument('--l4', type=int, default=1)
    args = parser.parse_args()

    x1, y1, x2, y2, x_test, y_test = load_intrusion()

    def sf(x, num_features):  # select features
        selected_idx = np.arange(num_features)
        return x[:, selected_idx]

    dataset2 = (x2, y2, x_test, y_test)

    dataset1 = (sf(x1, args.features), y1, sf(x_test, args.features), y_test)
    train(dataset1, dataset2, [args.l1, args.l2, args.l3, args.l4], if_restore=False)

    # dataset1 = (sf(x1, 80), y1, sf(x_test, 80), y_test)
    # train(dataset1, dataset2, [384, 96, 160, 96], if_restore=False)
    #
    # dataset1 = (sf(x1, 100), y1, sf(x_test, 100), y_test)
    # train(dataset1, dataset2, [64, 448, 192, 352], if_restore=False)