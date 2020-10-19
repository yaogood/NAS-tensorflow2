import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, callbacks
from global_parameters import *

def dense_model(layer_nerons):
    model = keras.models.Sequential()
    for node_i in layer_nerons:
        model.add(layers.Dense(node_i, activation='relu'))
    return model

def cnn_model(layer_architectures):
    model = keras.models.Sequential()
    for layer_arch in layer_architectures:
        filter_hw = layer_arch['filter_hw'] if 'filter_hw' in layer_arch.keys() else 3
        filter_c  = layer_arch['filter_c'] if 'filter_c' in layer_arch.keys() else 16
        stride_len = layer_arch['stride_len'] if 'stride_len' in layer_arch.keys() else 1
        padding    = layer_arch['padding'] if 'padding' in layer_arch.keys() else 'same'
        activation = layer_arch['activation'] if 'activation' in layer_arch.keys() else 'relu'
        model.add(
            layers.Conv2D(
                filter_c,
                (filter_hw, filter_hw),
                strides=(stride_len, stride_len),
                padding=padding,
                activation=activation
            )
        )
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(10, activation='softmax'))
    return model

def rnn_model(actions):
    pass

class Env:
    def __init__(self, dataset, epochs=100, child_batchsize=128, acc_beta=0.8, clip_rewards=0.0, env_type='Dense'):
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0
        self.type = env_type

        checkpoint_dir = 'child_nn_weights/' + data_set_name
        checkpoint_file_name = 'weights-best-{epoch:04d}.ckpt'
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    def get_rewards(self, actions):
        # generate a child architecture
        train_loader, test_loader = self.dataset
        train_iter = iter(train_loader)
        sample = next(train_iter)
        print('batch shape', sample[0].shape, sample[1].shape)
        output_size = len(tf.unique(sample[1]))

        if self.type == 'CNN':
            model = cnn_model(actions)
        elif self.type == 'RNN':
            model = rnn_model(actions)
        else:
            input_size = np.array(sample[0].shape[1:]).prod()
            model = dense_model(actions)
            model.add(layers.Dense(output_size))
            model.build(input_shape=(None, input_size))

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False),
            loss = keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        model.summary()

        # unpack the dataset
        ds, ds_test = self.dataset

        checkpoint = callbacks.ModelCheckpoint(
            self.checkpoint_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch'
        )

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model.save_weights(self.checkpoint_path.format(epoch=0))

        model.fit(ds,
            validation_data=ds_val,
            #batch_size=self.batchsize,
            epochs=self.epochs,
            callbacks=[checkpoint],
            verbose=2,  #one line per epoch
            initial_epoch=0
        )

        # load best performance epoch then evaluate the model
        model = keras.models.create_model()
        model.load_weights(self.checkpoint_path, by_name=True, skip_mismatch=False)
        loss, acc = model.evaluate(ds_test, verbose=2, batch_size=self.batchsize)

        # compute the reward
        reward = (acc - self.moving_acc)

        # if rewards are clipped, clip them in the range -0.05 to 0.05
        if self.clip_rewards:
            reward = np.clip(reward, -0.05, 0.05)

        # update moving accuracy with bias correction for 1st update
        if self.beta > 0.0 and self.beta < 1.0:
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0
            reward = np.clip(reward, -0.1, 0.1)
        print("Manager: EWA Accuracy = ", self.moving_acc)

        del model
        
        return reward, acc
