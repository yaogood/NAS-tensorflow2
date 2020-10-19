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
import tensorflow as tf
import tensorflow.keras as keras

from model import NASNetworkCIFAR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Basic model parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
parser.add_argument('--model_dir', type=str, default='models')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--cells', type=int, default=6)
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=36)
parser.add_argument('--cutout_size', type=int, default=8)
parser.add_argument('--grad_bound', type=float, default=10.0)
parser.add_argument('--initial_lr', type=float, default=0.025)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=9)
parser.add_argument('--train_from_scratch', type=bool, default=False)
args = parser.parse_args()

utils.create_exp_dir(args.model_dir)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

def train(train_ds, model, optimizer, global_step, criterion, classes=10):
    objs = utils.AvgMeter()
    top1 = utils.AvgMeter()
    top5 = utils.AvgMeter()

    for step, (input, labels) in enumerate(train_ds):
        global_step.assign_add(1)
        with tf.GradientTape() as tape:
            logits, aux_logits = model(input, global_step, training=True)
            loss = criterion(tf.one_hot(tf.squeeze(labels), depth=classes), logits)
            if aux_logits is not None:
                aux_loss = criterion(tf.one_hot(tf.squeeze(labels), depth=classes), aux_logits)
                loss += 0.4 * aux_loss
            reg_loss = args.l2_reg * tf.sqrt(
                tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in model.trainable_variables]))
            loss += reg_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        if args.grad_bound != 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, 15)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        ################################################################################################################
        acc1, acc5 = utils.accuracy(tf.nn.softmax(logits, axis=-1), tf.squeeze(labels), topk=(1, 5))
        batch_size = input.shape[0]
        objs.update(loss.numpy(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

        if (step + 1) % 100 == 0:
            print('train step {} loss {} top1 {} top5 {}'.format(step + 1, objs.avg, top1.avg, top5.avg))
            logging.info('train step %03d loss %e top1 %f top5 %f', step + 1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, global_step


def valid(valid_ds, model, criterion, classes=10):
    objs = utils.AvgMeter()
    top1 = utils.AvgMeter()
    top5 = utils.AvgMeter()

    for step, (input, labels) in enumerate(valid_ds):
        logits, _ = model(input, training=False)
        loss = criterion(tf.one_hot(tf.squeeze(labels), depth=classes), logits)

        acc1, acc5 = utils.accuracy(tf.nn.softmax(logits, axis=-1), tf.squeeze(labels), topk=(1, 5))
        batch_size = input.shape[0]
        objs.update(loss.numpy(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

        if (step + 1) % 100 == 0:
            print('valid step {} loss {} top1 {} top5 {}'.format(step + 1, objs.avg, top1.avg, top5.avg))
            logging.info('valid step %03d %e %f %f', step + 1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg



def train_cifar10():
    logging.info("Args = %s", args)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    epoch = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    best_acc_top1 = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

    ################################################ model setup #######################################################
    train_ds, test_ds = utils.load_cifar10(args.batch_size, args.cutout_size)
    total_steps = int(np.ceil(50000 / args.batch_size)) * args.epochs

    model = NASNetworkCIFAR(classes=10,
                            reduce_distance=args.cells,
                            num_nodes=args.nodes,
                            channels=args.channels,
                            keep_prob=args.keep_prob,
                            drop_path_keep_prob=args.drop_path_keep_prob,
                            use_aux_head=args.use_aux_head,
                            steps=total_steps,
                            arch=args.arch)

    temp_ = tf.random.uniform((64,32,32,3), minval=0, maxval=1, dtype=tf.float32)
    temp_ = model(temp_, step=1, training=True)
    model.summary()
    model_size = utils.count_parameters_in_MB(model)
    print("param size = {} MB".format(model_size))
    logging.info("param size = %fMB", model_size)

    criterion = keras.losses.CategoricalCrossentropy(from_logits=True)
    learning_rate = keras.experimental.CosineDecay(initial_learning_rate=args.initial_lr,
                                                   decay_steps=total_steps, alpha=0.0001)
    # learning_rate = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=args.initial_lr, decay_steps=total_steps, decay_rate=0.99, staircase=False, name=None
    # )
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)


    ########################################## restore checkpoint ######################################################
    if args.train_from_scratch:
        utils.clean_dir(args.model_dir)

    checkpoint_path = os.path.join(args.model_dir, 'checkpoints')
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer,
                               global_step=global_step,
                               epoch=epoch,
                               best_acc_top1=best_acc_top1)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    ############################################# training process #####################################################
    acc_train_result = []
    loss_train_result = []
    acc_test_result = []
    loss_test_result = []

    while epoch.numpy() < args.epochs:
        print('epoch {} lr {}'.format(epoch.numpy(), optimizer._decayed_lr(tf.float32)))

        train_acc, train_loss, step = train(train_ds, model, optimizer, global_step, criterion, classes=10)
        test_acc, test_loss = valid(test_ds, model, criterion, classes=10)

        acc_train_result.append(train_acc)
        loss_train_result.append(train_loss)
        acc_test_result.append(test_acc)
        loss_test_result.append(test_loss)

        logging.info('epoch %d lr %e', epoch.numpy(), optimizer._decayed_lr(tf.float32))
        logging.info(acc_train_result)
        logging.info(loss_train_result)
        logging.info(acc_test_result)
        logging.info(loss_test_result)

        is_best = False
        if test_acc > best_acc_top1:
            best_acc_top1 = test_acc
            is_best = True
        epoch.assign_add(1)
        if (epoch.numpy() + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch.numpy() + 1, ckpt_save_path))
        if is_best:
            pass

    utils.plot_single_list(acc_train_result, x_label='epochs', y_label='acc', file_name='acc_train')
    utils.plot_single_list(loss_train_result, x_label='epochs', y_label='loss', file_name='loss_train')
    utils.plot_single_list(acc_test_result, x_label='epochs', y_label='acc', file_name='acc_test')
    utils.plot_single_list(loss_test_result, x_label='epochs', y_label='loss', file_name='loss_test')

if __name__ == '__main__':
    import time
    start_time = time.time()
    train_cifar10()
    print("--- %s seconds ---" % (time.time() - start_time))

