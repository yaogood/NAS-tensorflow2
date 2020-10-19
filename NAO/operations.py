import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

BIAS=False

def apply_drop_path(x, drop_path_keep_prob, cell_id, num_cells, global_step, total_steps): ### ???
    # print(x.shape, drop_path_keep_prob, cell_id, num_cells, global_step, total_steps)
    layer_ratio = float(cell_id + 1) / float(num_cells)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(global_step + 1) / float(total_steps)
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

    if drop_path_keep_prob < 1.0:
        # mask = torch.FloatTensor(x.shape[0], 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        mask_shape = (x.shape[0], 1, 1, x.shape[-1])
        mask_bool = tf.random.uniform(mask_shape) < drop_path_keep_prob
        mask = tf.where(mask_bool, tf.ones(mask_shape), tf.zeros(mask_shape)) # bernoulli distribution
        x = x / drop_path_keep_prob * mask
    return x


class AuxHeadCIFAR(keras.Model):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()
        self.relu1 = layers.ReLU()
        self.avg_pool = layers.AveragePooling2D(pool_size=3, strides=1, padding='valid')
        self.conv1 = layers.Conv2D(filters=128, kernel_size=1, use_bias=BIAS)
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters=768, kernel_size=2, use_bias=BIAS)
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu3 = layers.ReLU()
        self.classifier = layers.Dense(classes)

    def call(self, x, training=False, **kwargs):
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu3(x)
        aux_logits = self.classifier(tf.reshape(x, [x.shape[0], -1]))
        return aux_logits


class AuxHeadImageNet(keras.Model):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = layers.ReLU()
        self.avg_pool = layers.AveragePooling2D(pool_size=5, strides=2, padding='valid')
        self.conv1 = layers.Conv2D(filters=128, kernel_size=1, use_bias=BIAS)
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters=768, kernel_size=2, use_bias=BIAS)
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu3 = layers.ReLU()
        self.classifier = layers.Dense(classes)

    def call(self, x, training=False, **kwargs):
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu3(x)
        # x = self.classifier(x.view(x.shape[0], -1))
        x = self.classifier(tf.reshape(x, [x.shape[0], -1]))
        return x


class Conv(keras.Model):
    def __init__(self, C_in, C_out, kernel_size, strides, padding):
        super(Conv, self).__init__()
        self.padding = padding
        if isinstance(kernel_size, int):
            self.ops = keras.Sequential([
                layers.ReLU(),
                layers.Conv2D(filters=C_out, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=BIAS),
                layers.BatchNormalization(axis=-1)
            ])
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = keras.Sequential([
                layers.ReLU(),
                layers.Conv2D(filters=C_out, kernel_size=(k1, k2), strides=(1, strides), padding=padding, use_bias=BIAS),
                layers.BatchNormalization(axis=-1),
                layers.ReLU(),
                layers.Conv2D(filters=C_out, kernel_size=(k2, k1), strides=(strides, 1), padding=padding, use_bias=BIAS),
                layers.BatchNormalization(axis=-1),
            ])

    def call(self, x, training=False, **kwargs):
        x = self.ops(x, training=training)
        return x


class SepConv(keras.Model):
    def __init__(self, C_in, C_out, kernel_size, strides, padding):
        super(SepConv, self).__init__() # strides=1 firstly or strides=strides firstly ???
        self.op = keras.Sequential([
            layers.ReLU(),
            layers.Conv2D(C_in, kernel_size=kernel_size, strides=strides, padding=padding, groups=C_in, use_bias=BIAS),
            layers.Conv2D(C_in, kernel_size=1, use_bias=BIAS),
            layers.BatchNormalization(axis=-1),
            layers.ReLU(),
            layers.Conv2D(C_in, kernel_size=kernel_size, strides=1, padding=padding, groups=C_in, use_bias=BIAS),
            layers.Conv2D(C_out, kernel_size=1, use_bias=BIAS),
            layers.BatchNormalization(axis=-1),
        ])

    def call(self, x, training=False, **kwargs):
        return self.op(x, training=training)


class Identity(keras.Model):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, **kwargs):
        return x

class ReLUConvBN(keras.Model):
    def __init__(self, C_in, C_out, kernel_size, strides, padding):
        super(ReLUConvBN, self).__init__()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(filters=C_out, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=BIAS)
        self.bn = layers.BatchNormalization(axis=-1)

    def call(self, x, training=False, **kwargs):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x

class FactorizedReduce(keras.Model):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.path1 = keras.Sequential([layers.AveragePooling2D(pool_size=1, strides=2, padding='valid'),
                                   layers.Conv2D(filters=C_out // 2, kernel_size=1, use_bias=False)])
        self.path2 = keras.Sequential([layers.AveragePooling2D(pool_size=1, strides=2, padding='valid'),
                                   layers.Conv2D(filters=C_out // 2, kernel_size=1, use_bias=False)])
        self.bn = layers.BatchNormalization(axis=-1)

    def call(self, x, training=False, **kwargs):
        x_path1 = x  # notice here we choose from 2-th element of the pixels, the last dim is the number of channels.
        x_path2 = tf.pad(x[:, 1:, 1:, :], [[0, 0], [0, 1], [0, 1], [0, 0]], mode="constant", constant_values=0)
        x_path1 = self.path1(x_path1)
        x_path2 = self.path2(x_path2)
        out = tf.concat([x_path1, x_path2], axis=-1)
        out = self.bn(out, training=training)
        return out


class CalibrateSize(keras.Model):
    def __init__(self, in_shapes, channels):
        super(CalibrateSize, self).__init__()
        self.channels = channels
        hw = [shape[0] for shape in in_shapes]
        c = [shape[-1] for shape in in_shapes]

        x_shape = [hw[0], hw[0], c[0]]
        y_shape = [hw[1], hw[1], c[1]]

        # previous reduction cell
        if x_shape[0] != y_shape[0]:
            assert x_shape[0] == 2 * y_shape[0]
            self.relu = layers.ReLU()
            self.preprocess_x = FactorizedReduce(x_shape[-1], channels)
            x_shape = [hw[1], hw[1], channels]

        if x_shape[-1] != channels:
            self.preprocess_x = ReLUConvBN(x_shape[-1], channels, kernel_size=1, strides=1, padding='same')
            x_shape[-1] = channels

        if y_shape[-1] != channels:
            self.preprocess_y = ReLUConvBN(y_shape[-1], channels, kernel_size=1, strides=1, padding='same')
            y_shape[-1] = channels

        self.out_shapes = [x_shape, y_shape]

    def call(self, x, y=None, training=False, **kwargs):
        if x.shape[2] != y.shape[2]:
            x = self.relu(x)
            x = self.preprocess_x(x, training=training)
        if x.shape[-1] != self.channels:
            x = self.preprocess_x(x, training=training)
        if y.shape[-1] != self.channels:
            y = self.preprocess_y(y, training=training)
        return x, y


class FinalCombine(keras.Model):
    def __init__(self, out_shapes, out_hw, channels, concat_index):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat_index = concat_index
        self.ops = []
        self.concat_fac_op_dict = {}
        for i in concat_index:
            hw = out_shapes[i][0]
            if hw > out_hw:
                assert hw == 2 * out_hw and i in [0, 1]
                self.concat_fac_op_dict[i] = len(self.ops)
                self.ops.append(FactorizedReduce(out_shapes[i][-1], channels))

    def call(self, states, training=False, **kwargs):
        for i in self.concat_index:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i], training)
        out = tf.concat([states[i] for i in self.concat_index], axis=-1)
        return out

########################################################################################################################

class WSAvgPool2d(keras.Model):
    def __init__(self, kernel_size, padding):
        super(WSAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def call(self, x, strides=1, **kwargs):
        return tf.nn.avg_pool2d(x, ksize=self.kernel_size, strides=strides, padding=self.padding)


class WSMaxPool2d(keras.Model):
    def __init__(self, kernel_size, padding):
        super(WSMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def call(self, x, strides=1, **kwargs):
        return tf.nn.max_pool2d(x, ksize=self.kernel_size, strides=strides, padding=self.padding)


class WSReLUConvBN(keras.Model):
    def __init__(self, num_possible_inputs, C_out, C_in, kernel_size, strides=1, padding='valid'):
        super(WSReLUConvBN, self).__init__()
        self.strides = strides
        self.padding = padding
        self.relu = layers.ReLU()
        self.w = []
        for _ in range(num_possible_inputs):
            self.w.append(tf.Variable(tf.random.normal((kernel_size, kernel_size, C_in, C_out), dtype=tf.float32)))
        self.bn = layers.BatchNormalization(axis=-1)

    def forward(self, x, x_id, training=False):
        x = self.relu(x)
        w = tf.concat([self.w[i] for i in x_id], axis=2)
        x = tf.nn.conv2d(x, w, strides=self.strides, padding=self.padding)
        x = self.bn(x, training=training)
        return x


class WSBN(keras.Model):

    def __init__(self, num_possible_inputs, num_features, epsilon=1e-5, momentum=0.1, trainable=True):
        super(WSBN, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.trainable = trainable
        self.weight = None
        self.bias = None
        if self.trainable:
            self.weight = []
            self.bias = []
            for _ in range(num_possible_inputs):
                self.weight.append(tf.Variable(tf.random.uniform(num_features)))
                self.bias.append(tf.Variable(tf.random.uniform(num_features)))
        self.running_mean = tf.Variable(tf.zeros(num_features))
        self.running_var = tf.Variable(tf.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean = tf.Variable(tf.zeros(self.num_features))
        self.running_var = tf.Variable(tf.ones(self.num_features))
        if self.trainable:
            for i in range(self.num_possible_inputs):
                self.weight[i] = tf.Variable(tf.ones_like(self.weight[i]))
                self.bias[i] = tf.Variable(tf.zeros_like(self.weight[i]))

    def forward(self, x, x_id, training=False):
        # return F.batch_norm(
            # x, self.running_mean, self.running_var, self.weight[x_id], self.bias[x_id],
            # training, self.momentum, self.eps)
        return tf.nn.batch_normalization(x, self.running_mean, self.running_var,
                                         self.bias[x_id], self.weight[x_id], self.epsilon)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' trainable={trainable})'.format(name=self.__class__.__name__, **self.__dict__))


class WSSepConv(keras.Model):
    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(WSSepConv, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding

        self.relu1 = layers.ReLU()
        self.W1_depthwise = []
        self.W1_pointwise = []
        for i in range(num_possible_inputs):
            self.W1_depthwise.append(tf.Variable(tf.random.normal((kernel_size, kernel_size, C_in, 1), dtype=tf.float32)))
            self.W1_pointwise.append(tf.Variable(tf.random.normal((1, 1, C_in, C_in), dtype=tf.float32)))
        self.W1_pointwise = nn.ParameterList(
            [nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn1 = WSBN(num_possible_inputs, C_in, affine=affine)

        self.relu2 = nn.ReLU(inplace=INPLACE)
        self.W2_depthwise = nn.ParameterList(
            [nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W2_pointwise = nn.ParameterList(
            [nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn2 = WSBN(num_possible_inputs, C_in, affine=affine)

    def forward(self, x, x_id, stride=1, bn_train=False):
        x = self.relu1(x)
        x = F.conv2d(x, self.W1_depthwise[x_id], stride=stride, padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W1_pointwise[x_id], padding=0)
        x = self.bn1(x, x_id, bn_train=bn_train)

        x = self.relu2(x)
        x = F.conv2d(x, self.W2_depthwise[x_id], padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W2_pointwise[x_id], padding=0)
        x = self.bn2(x, x_id, bn_train=bn_train)
        return x