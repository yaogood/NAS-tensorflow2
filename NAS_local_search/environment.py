import numpy as np
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import sklearn.metrics as sm

import os
if not os.path.exists('weights/'):
    os.makedirs('weights/')

class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, epochs=5, child_batchsize=128, acc_beta=0.8, clip_rewards=0.0):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: float - to clip rewards in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

    def get_rewards(self, actions):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given actions
        '''
        with tf.compat.v1.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)
            # unpack the dataset
            X_train, y_train, X_val, y_val = self.dataset
            # generate a submodel given predicted actions
            model = dense_model(actions)
            model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
            model.build(input_shape=(None, X_train.shape[1]))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            checkpoint_path = 'weights-best.hdf5'
            checkpoint = ModelCheckpoint(checkpoint_path,
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=True)
            callbacks_list = [checkpoint]

            # train the model using Keras methods
            model.fit(X_train, y_train,
                    validation_split=0.12,
                    batch_size=self.batchsize,
                    epochs=self.epochs,
                    callbacks=callbacks_list,
                    verbose=0)

            # load best performance epoch in this training session
            model.load_weights('weights-best.hdf5')

            # evaluate the model
            loss, acc = model.evaluate(X_val, y_val, batch_size=self.batchsize, verbose=0)
            # predicted = model.predict(X_val)
            # p = tf.argmax(predicted, axis=1)
            # a = tf.argmax(y_val, axis=1)
            # p = p.to_numpy()
            # a = a.to_numpy()
            # acc = sm.f1_score(p.eval(), a.eval(), average='macro') # acc is f1 score in this experiment
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

        # clean up resources and GPU memory
        network_sess.close()

        return reward, acc, self.moving_acc

def dense_model(layer_nerons):
    model = keras.models.Sequential()
    for node_i in layer_nerons:
        model.add(keras.layers.Dense(node_i, activation='relu'))
    return model