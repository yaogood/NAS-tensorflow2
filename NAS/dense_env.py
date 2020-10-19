import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, callbacks
from load_data import load_covertype, load_sensit, load_intrusion
import sklearn.metrics as sm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(9)
np.random.seed(9)

def dense_model(layer_nerons):
    model = keras.models.Sequential()
    for node_i in layer_nerons:
        model.add(layers.Dense(node_i, activation='relu'))
    return model

class DenseEnv:
    def __init__(self, dataset, epochs=100, batchsize=128):
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = batchsize

        checkpoint_dir = 'dense_weights/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_file_name = 'weights-best.hdf5'
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file_name)

    def get_rewards(self, actions):
        # generate a child architecture
        x, y, x_test, y_test = self.dataset
        # print("current actions: ", actions)
        model = dense_model(actions)
        model.add(layers.Dense(y.shape[1]))
        model.build(input_shape=(None, x.shape[1]))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        # model.summary()
        my_callback = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min', min_delta=0.0001),
            callbacks.ModelCheckpoint(self.checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
                                      save_weights_only=True, mode='min', save_freq='epoch')
        ]

        model.fit(
            x,y,
            # validation_data=(x_test,y_test),
            validation_split=0.12,
            batch_size=self.batchsize,
            epochs=self.epochs,
            callbacks=my_callback,
            verbose=2,  #0 = silent, 1 = progress bar, 2 = one line per epoch
            initial_epoch=0
        )

        # load best performance epoch then evaluate the model
        model.load_weights(self.checkpoint_path)
        loss, acc = model.evaluate(x_test, y_test, verbose=2, batch_size=self.batchsize)
        # print("\nchild architecture test loss:", loss)
        print("child architecture test accuracy", acc)

        predicted = model.predict(x_test)
        p = tf.argmax(predicted, axis=1)
        a = tf.argmax(y_test, axis=1)
        f1 = sm.f1_score(p, a, average='macro')
        print("f1 score macro:", f1)
        # print("f1 score micro:", sm.f1_score(p, a, average='micro'))
        # print("f1 score weighted:", sm.f1_score(p, a, average='weighted'))

        return acc # return f1

# def get_f1(predicted, actual):
#     TP = tf.math.count_nonzero(predicted * actual)
#     TN = tf.math.count_nonzero((predicted - 1) * (actual - 1))
#     FP = tf.math.count_nonzero(predicted * (actual - 1))
#     FN = tf.math.count_nonzero((predicted - 1) * actual)
#
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1 = 2 * precision * recall / (precision + recall)
#     return f1

if __name__ == '__main__':
    # x1, y1, x2, y2, x_test, y_test = load_intrusion()
    import time
    start_time = time.time()

    while (1):
        x1, y1, x2, y2, x_test, y_test = load_covertype()
        dataset = (x1, y1, x_test, y_test)
        env = DenseEnv(dataset, epochs=200, batchsize=128)
        # array([[60, 45, 53, 61]])
        r1 = env.get_rewards([296, 336, 368, 312])
        print(r1)
        break
        # dataset = (x2, y2, x_test, y_test)
        # env = DenseEnv(dataset, epochs=5, batchsize=128)
        # r2 = env.get_rewards([512, 512, 512, 512])
        #
        # if abs(r1 - r2) < 0.02:
        #     break

    print("--- %s seconds ---" % (time.time() - start_time))
