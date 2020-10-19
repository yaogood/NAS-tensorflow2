import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorboardX import SummaryWriter
from tensorflow.keras import Sequential, layers, optimizers, datasets

from global_parameters import parse_args
from dense_env import DenseEnv
from load_data import load_structure_dataset, load_covertype, load_sensit, load_intrusion
from search_space import SearchSpaceDense
from nao import NAO, train_nao

print(tf.version.VERSION)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

checkpoint_dir = 'controller_weights/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

args = parse_args()
# construct a state space
ss = SearchSpaceDense(num_layers=args.num_layers)
ss.add_state(name='neurons', values=range(32, 512, 32))
ss.print_search_space()


def train(dataset, epochs, total_steps, save_filename=None, load_filename=None, hidden=None, nao=None):
    if not nao:
        print("\ninitial model weights successfully.\n")
        nao = NAO(search_space=ss,
                         embedding_size=args.embedding_size,
                         encoder_hidden_size=args.encoder_hidden_size,
                         encoder_dropout=args.encoder_dropout,
                         mlp_num_layers=args.mlp_num_layers,
                         mlp_hidden_size=args.mlp_hidden_size,
                         mlp_dropout=args.mlp_dropout,
                         decoder_hidden_size=args.decoder_hidden_size,
                         decoder_dropout=args.decoder_dropout)
    if load_filename:
        print("\nload ", load_filename, " model weights successfully.")
        checkpoint_path = os.path.join(checkpoint_dir, load_filename)
        nao.load_weights(checkpoint_path)

    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.03, decay_steps=100,
                                                        decay_rate=0.95, staircase=True)
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    env = DenseEnv(dataset, epochs=epochs, batchsize=args.dataset_batch_size)

    # train the controller on the saved state and the discounted rewards
    arch_buffer, accuracy_buffer, ema_reward_buffer, loss_buffer, hidden \
        = train_nao(nao,
                    env,
                    nao_dataset=nao_dataset,
                    search_space=ss,
                    epochs=args.epochs,
                    optimizer=optimizer,
                    alpha=args.alpha)

    best_archs = nao(None, training=False, prev_hidden=hidden)
    best_actions = None
    best_acc = 0.0
    for arch in best_archs:
        actions = ss.arch_list_to_values(arch.numpy())
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
        nao.save_weights(checkpoint_path)

    return hidden, nao


if __name__ == '__main__':
    pass
