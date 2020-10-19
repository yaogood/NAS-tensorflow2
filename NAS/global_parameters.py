import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='networkIntrusion', help="Supported datasets: MNIST, CIFAR10, CIFAR100, coverType, networkIntrusion, sensIT.")
    parser.add_argument('--network_type', type=str, default='Dense', help="Supported networks: Dense, CNN, RNN.")
    # parser.add_argument('--partial_features', type=float, default=1.0, help="The ratio of partial features, range(0.0, 1.0]")
    parser.add_argument('--embedding_dim', type=int, default=20, help='Dimentionality of input embedding.')
    parser.add_argument('--lstm_units', type=int, default=64, help='Dimentionality of hidden state in LSTM controller.')
    parser.add_argument('--controller_batch_size', type=int, default=1, help='Number of Monte Carlo samples in each step.')
    parser.add_argument('--reg_strength', type=float, default=1e-3, help='Regularation stength when training controller.')
    parser.add_argument('--entropy_coeff', type=float, default=1e-3, help='Entropy coeff for entropy loss.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in child architectures.')
    # parser.add_argument('--epochs', type=int, default=10, help='Epochs when training child architectures.')
    parser.add_argument('--dataset_batch_size', type=int, default=128, help='Dataset batch size when training child architectures.')
    parser.add_argument('--ema_alpha', type=float, default=0.15, help='Exponential Moving Accuracy/Reward for reducing variances.')
    # parser.add_argument('--total_steps',type=int, default=1000, help='Total steps in an episode when doing RL')
    parser.add_argument('--clip_norm', type=float, default=15.0, help='Clip by L2 norm of controller weights\' gradient')
    parser.add_argument('--more_exploration', type=str2bool, default=False, help='Explore more by adding entropy loss.')
    parser.add_argument('--with_regularation', type=str2bool, default=False, help='Use regularation when training controller.')
    parser.add_argument('--load_path', type=str, help='Reload model path for resuming training')
    args = parser.parse_args()
    return args