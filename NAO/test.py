import tensorflow.keras as keras

(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(type(x))