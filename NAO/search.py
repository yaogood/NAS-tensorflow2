import os
import sys
import glob
import time
import copy
import random
import numpy as np
import utils
import logging
import argparse
from model import NASNetworkCIFAR
# from model_search import NASWSNetworkCIFAR, NASWSNetworkImageNet
from nao import NAO

# Basic model parameters.
parser = argparse.ArgumentParser(description='NAO CIFAR-10')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100, imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--child_batch_size', type=int, default=64)
parser.add_argument('--child_eval_batch_size', type=int, default=500)
parser.add_argument('--child_epochs', type=int, default=150)
parser.add_argument('--child_layers', type=int, default=3)
parser.add_argument('--child_nodes', type=int, default=5)
parser.add_argument('--child_channels', type=int, default=20)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.001)
parser.add_argument('--child_keep_prob', type=float, default=1.0)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.9)
parser.add_argument('--child_l2_reg', type=float, default=3e-4)
parser.add_argument('--child_use_aux_head', action='store_true', default=False)
parser.add_argument('--child_eval_epochs', type=str, default='30')
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--child_lr', type=float, default=0.1)
parser.add_argument('--child_label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--child_gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--child_decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--controller_seed_arch', type=int, default=600)
parser.add_argument('--controller_expand', type=int, default=None)
parser.add_argument('--controller_new_arch', type=int, default=300)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_encoder_emb_size', type=int, default=48)
parser.add_argument('--controller_mlp_layers', type=int, default=3)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_source_length', type=int, default=40)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=40)
parser.add_argument('--controller_encoder_dropout', type=float, default=0)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0)
parser.add_argument('--controller_l2_reg', type=float, default=1e-4)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_bound', type=float, default=5.0)
args = parser.parse_args()


def get_builder(dataset):
    if dataset == 'cifar10':
        return build_cifar10
    # elif dataset == 'cifar100':
    #     return build_cifar100
    # else:
    #     return build_imagenet


def build_cifar10(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_ds, valid_ds = utils.load_cifar10(args.child_batch_size, args.child_cutout_size) # subset random sample

    model = NASWSNetworkCIFAR(10, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob,
                              args.child_drop_path_keep_prob,
                              args.child_use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler

def main():
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.steps = int(np.ceil(45000 / args.child_batch_size)) * args.child_epochs

    logging.info("args = %s", args)

    if args.child_arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.child_arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_arch, archs))
            child_arch_pool = archs
    elif os.path.exists(os.path.join(args.output_dir, 'arch_pool')):
        logging.info('Architecture pool is founded, loading')
        with open(os.path.join(args.output_dir, 'arch_pool')) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_arch, archs))
            child_arch_pool = archs
    else:
        child_arch_pool = None

    child_eval_epochs = eval(args.child_eval_epochs)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(ratio=0.9, epoch=-1)

    nao = NAO(
        args.controller_encoder_layers,
        args.controller_encoder_vocab_size,
        args.controller_encoder_hidden_size,
        args.controller_encoder_dropout,
        args.controller_encoder_length,
        args.controller_source_length,
        args.controller_encoder_emb_size,
        args.controller_mlp_layers,
        args.controller_mlp_hidden_size,
        args.controller_mlp_dropout,
        args.controller_decoder_layers,
        args.controller_decoder_vocab_size,
        args.controller_decoder_hidden_size,
        args.controller_decoder_dropout,
        args.controller_decoder_length,
    )
    nao = nao.cuda()
    logging.info("Encoder-Predictor-Decoder param size = %fMB", utils.count_parameters_in_MB(nao))

    # Train child model
    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_seed_arch, args.child_nodes, 5)  # [[[conv],[reduc]]]
    child_arch_pool_prob = None

    eval_points = utils.generate_eval_points(child_eval_epochs, 0, args.child_epochs)
    step = 0
    for epoch in range(1, args.child_epochs + 1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        # sample an arch to train
        train_acc, train_obj, step = child_train(train_queue, model, optimizer, step, child_arch_pool, child_arch_pool_prob, train_criterion)
        logging.info('train_acc %f', train_acc)

        if epoch not in eval_points:
            continue
        # Evaluate seed archs
        valid_accuracy_list = child_valid(valid_queue, model, child_arch_pool, eval_criterion)

        # Output archs and evaluated error rate
        old_archs = child_arch_pool
        old_archs_perf = valid_accuracy_list

        old_archs_sorted_indices = np.argsort(old_archs_perf)[::-1]
        old_archs = [old_archs[i] for i in old_archs_sorted_indices]
        old_archs_perf = [old_archs_perf[i] for i in old_archs_sorted_indices]
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(epoch)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(epoch)), 'w') as fp:
                with open(os.path.join(args.output_dir, 'arch_pool'), 'w') as fa_latest:
                    with open(os.path.join(args.output_dir, 'arch_pool.perf'), 'w') as fp_latest:
                        for arch, perf in zip(old_archs, old_archs_perf):
                            arch = ' '.join(map(str, arch[0] + arch[1]))
                            fa.write('{}\n'.format(arch))
                            fa_latest.write('{}\n'.format(arch))
                            fp.write('{}\n'.format(perf))
                            fp_latest.write('{}\n'.format(perf))

        if epoch == args.child_epochs:
            break

        # Train Encoder-Predictor-Decoder
        logging.info('Training Encoder-Predictor-Decoder')
        encoder_input = list \
            (map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), old_archs))
        # [[conv, reduc]]
        min_val = min(old_archs_perf)
        max_val = max(old_archs_perf)
        encoder_target = [(i - min_val) / (max_val - min_val) for i in old_archs_perf]

        if args.controller_expand:
            dataset = list(zip(encoder_input, encoder_target))
            n = len(dataset)
            ratio = 0.9
            split = int( n *ratio)
            np.random.shuffle(dataset)
            encoder_input, encoder_target = list(zip(*dataset))
            train_encoder_input = list(encoder_input[:split])
            train_encoder_target = list(encoder_target[:split])
            valid_encoder_input = list(encoder_input[split:])
            valid_encoder_target = list(encoder_target[split:])
            for _ in range(args.controller_expan d -1):
                for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                    a = np.random.randint(0, args.child_nodes)
                    b = np.random.randint(0, args.child_nodes)
                    src = src[:4 * a] + src[4 * a + 2:4 * a + 4] + \
                          src[4 * a:4 * a + 2] + src[4 * (a + 1):20 + 4 * b] + \
                          src[20 + 4 * b + 2:20 + 4 * b + 4] + src[20 + 4 * b:20 + 4 * b + 2] + \
                          src[20 + 4 * (b + 1):]
                    train_encoder_input.append(src)
                    train_encoder_target.append(tgt)
        else:
            train_encoder_input = encoder_input
            train_encoder_target = encoder_target
            valid_encoder_input = encoder_input
            valid_encoder_target = encoder_target
        logging.info('Train data: {}\tValid data: {}'.format(len(train_encoder_input), len(valid_encoder_input)))

        nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True, swap=True if args.controller_expand is None else False)
        nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
        nao_train_queue = torch.utils.data.DataLoader(
            nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
        nao_valid_queue = torch.utils.data.DataLoader(
            nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
        nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
        for nao_epoch in range(1, args.controller_epoch s +1):
            nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, nao, nao_optimizer)
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
            if nao_epoch % 100 == 0:
                pa, hs = nao_valid(nao_valid_queue, nao)
                logging.info("Evaluation on valid data")
                logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', epoch, pa, hs)

        # Generate new archs
        new_archs = []
        max_step_size = 50
        predict_step_size = 0
        top100_archs = list \
            (map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), old_archs[:100]))
        nao_infer_dataset = utils.NAODataset(top100_archs, None, False)
        nao_infer_queue = torch.utils.data.DataLoader(
            nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
        while len(new_archs) < args.controller_new_arch:
            predict_step_size += 1
            logging.info('Generate new architectures with step size %d', predict_step_size)
            new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
            for arch in new_arch:
                if arch not in encoder_input and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > max_step_size:
                break
                # [[conv, reduc]]
        new_archs = list(map(lambda x: utils.parse_seq_to_arch(x, 2), new_archs))  # [[[conv],[reduc]]]
        num_new_archs = len(new_archs)
        logging.info("Generate %d new archs", num_new_archs)
        # replace bottom archs
        new_arch_pool = old_archs[:len(old_archs) - num_new_archs] + new_archs
        logging.info("Totally %d architectures now to train", len(new_arch_pool))

        child_arch_pool = new_arch_pool
        with open(os.path.join(args.output_dir, 'arch_pool'), 'w') as f:
            for arch in new_arch_pool:
                arch = ' '.join(map(str, arch[0] + arch[1]))
                f.write('{}\n'.format(arch))


        child_arch_pool_prob = None

    logging.info('Finish Searching')
    logging.info('Reranking top 5 architectures')
    # reranking top 5
    top_archs = old_archs[:5]
    if args.dataset == 'cifar10':
        top_archs_perf = train_and_evaluate_top_on_cifar10(top_archs, train_queue, valid_queue)
    elif args.dataset == 'cifar100':
        top_archs_perf = train_and_evaluate_top_on_cifar100(top_archs, train_queue, valid_queue)
    else:
        top_archs_perf = train_and_evaluate_top_on_imagenet(top_archs, train_queue, valid_queue)
    top_archs_sorted_indices = np.argsort(top_archs_perf)[::-1]
    top_archs = [top_archs[i] for i in top_archs_sorted_indices]
    top_archs_perf = [top_archs_perf[i] for i in top_archs_sorted_indices]
    with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as fa:
        with open(os.path.join(args.output_dir, 'arch_pool.perf.final'), 'w') as fp:
            for arch, perf in zip(top_archs, top_archs_perf):
                arch = ' '.join(map(str, arch[0] + arch[1]))
                fa.write('{}\n'.format(arch))
                fp.write('{}\n'.format(perf))