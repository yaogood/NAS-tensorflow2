import os
import shutil
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def plot_single_list(data, x_label=None, y_label=None, line_label=None, title=None,
                     x_min=None, x_max=None, y_min=None, y_max=None, file_name=None):
    plt.figure()
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if not line_label:
        line_label = y_label
    plt.plot(range(len(data)), data, label=line_label)
    if x_min and x_max and y_min and y_max:
        plt.axis([x_min, x_max, y_min, y_max])
        plt.legend()
    if not file_name:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = './images/' + timestr + '.png'
    else:
        file_name = './images/' + file_name + '.png'
    plt.title(title)
    plt.savefig(file_name)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * (100.0 / batch_size))
        res.append(acc)
    return res

def pairwise_accuracy(la, lb):
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    N = len(la)
    assert N == len(lb)

    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c

    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N

def clean_dir(DIR_PATH):
    if tf.io.gfile.exists(DIR_PATH):
        print('Removing existing dir: {}'.format(DIR_PATH))
        tf.io.gfile.rmtree(DIR_PATH)

def create_dir(DIR_PATH, scripts_to_save=None):
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
        print('Creating dir : {}'.format(DIR_PATH))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(DIR_PATH, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(DIR_PATH, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


########################################################################################################################
# cifar10 dataset
IMG_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def preprocessing(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    image = (image / 255.0)
    return image, label

def augment(image,label):
    image, label = preprocessing(image, label)
    # padding, crop
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4) # Add 4 pixels of padding
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3]) # Random crop back to the original size
    image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image, 0, 1)
    return image, label

def normalize(image, label):
    # this function normalize inputs for zero mean and unit variance. It is used when training a model.
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    img = tf.transpose(image, [2,0,1])
    res = []
    for c in range(img.shape[0]):
        res.append((img[c] - CIFAR_MEAN[c]) / (CIFAR_STD[c] + 1e-7))
    return tf.transpose(tf.stack(res), [1,2,0]), label

def cutout(image, mask_size):
    h, w = image.shape[0], image.shape[1]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = tf.convert_to_tensor(mask)
    mask = tf.transpose(tf.broadcast_to(mask, [3, h, w]), [1,2,0])
    image *= mask
    return image

# def cutout(image, cutout_size):
#     offset_h = np.random.randint(image.shape[0])
#     offset_w = np.random.randint(image.shape[1])
#     tfa.image.cutout(image, cutout_size, (offset_h, offset_w), constant_values=0)

def load_cifar10(batch_size, cutout_size):
    print('loading data...')
    (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # (train_loader, test_loader), ds_info = tfds.load('cifar10', split=['train', 'test'],
    #                                                  as_supervised=True,
    #                                                  with_info=True)
    train_loader = tf.data.Dataset.from_tensor_slices((x, y))
    train_loader = train_loader.map(augment, num_parallel_calls=AUTOTUNE)
    train_loader = train_loader.map(normalize)
    if cutout_size is not None:
        train_loader = train_loader.map(lambda x,y: (cutout(x, cutout_size), y))
    train_loader = train_loader.shuffle(5000).batch(batch_size).prefetch(AUTOTUNE)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.map(preprocessing, num_parallel_calls=AUTOTUNE)
    test_loader = test_loader.map(normalize)
    test_loader = test_loader.batch(batch_size).prefetch(AUTOTUNE)

    print('done.')
    return train_loader, test_loader

def show_img(ds):
    image, label = next(iter(ds))
    plt.imshow(image[2])
    plt.show()
########################################################################################################################

def count_parameters_in_MB(model):
    # return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])/1e6

def build_arch(arch):
    if arch is None:
        return None, None
    # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
    arch = list(map(int, arch.strip().split()))
    length = len(arch)
    conv_dag = arch[:length//2]
    reduc_dag = arch[length//2:]
    return conv_dag, reduc_dag

# def archs_dataset(search_space, batch_size=1, max_num=200, sos_id=0, eos_id=0):
#     "generate the child architectures as dataset"
#
#     encoder_input = search_space.generate_random_arch()
#     decoder_input = [sos_id] + encoder_input[:-1]
#     sample = {
#         'encoder_input': tf.constant(encoder_input),
#         'decoder_input': tf.constant(decoder_input),
#         'decoder_labels': tf.constant(encoder_input),
#     }
#
#     return sample

# train_ds, test_ds = load_cifar10(128)


# train_loader, test_loader = load_cifar10(128, 8)
# show_img(train_loader)