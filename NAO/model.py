import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from operations import ReLUConvBN, CalibrateSize, FactorizedReduce, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path, FinalCombine
from operations import SepConv, Identity, Conv

# OPERATIONS = {
#     0: SepConv,  # 3x3
#     1: SepConv,  # 5x5
#     2: layers.AveragePooling2D, # 3x3
#     3: layers.MaxPool2D, # 3x3
#     4: Identity,
# }
#
# OPERATIONS_large = {
#     5: Identity,
#     6: Conv,  # 1x1
#     7: Conv,  # 3x3
#     8: Conv,  # 1x3 + 3x1
#     9: Conv,  # 1x7 + 7x1
#     10: layers.MaxPool2D,  # 2x2
#     11: layers.MaxPool2D,  # 3x3
#     12: layers.MaxPool2D,  # 5x5
#     13: layers.AveragePooling2D,  # 2x2
#     14: layers.AveragePooling2D,  # 3x3
#     15: layers.AveragePooling2D,  # 5x5
# }

def op(node_id, op_id, x_shape, channels, strides): # x means feature maps
    br_op = None
    x_id_fact_reduce = None

    x_stride = strides if node_id in [0, 1] else 1  ## ??? why set strides=1 when x_id not in [0, 1]

    if op_id == 0:
        br_op = SepConv(C_in=channels, C_out=channels, kernel_size=3, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 1:
        br_op = SepConv(C_in=channels, C_out=channels, kernel_size=5, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 2:
        br_op = layers.AveragePooling2D(pool_size=3, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, x_shape[-1]]
    elif op_id == 3:
        br_op = layers.MaxPool2D(pool_size=3, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, x_shape[-1]]
    elif op_id == 4:
        br_op = Identity()
        if x_stride > 1:
            assert x_stride == 2
            x_id_fact_reduce = FactorizedReduce(C_in=x_shape[-1], C_out=channels)
            x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 5:
        br_op = Identity()
        if x_stride > 1:
            assert x_stride == 2
            x_id_fact_reduce = FactorizedReduce(C_in=x_shape[-1], C_out=channels)
            x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 6:
        br_op = Conv(C_in=channels, C_out=channels, kernel_size=1, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 7:
        br_op = Conv(C_in=channels, C_out=channels, kernel_size=3, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 8:
        br_op = Conv(C_in=channels, C_out=channels, kernel_size=(1, 3), strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 9:
        br_op = Conv(C_in=channels, C_out=channels, kernel_size=(1, 7), strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 10:
        br_op = layers.MaxPool2D(pool_size=2, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 11:
        br_op = layers.MaxPool2D(pool_size=3, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 12:
        br_op = layers.MaxPool2D(pool_size=5, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 13:
        br_op = layers.AveragePooling2D(pool_size=2, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 14:
        br_op = layers.AveragePooling2D(pool_size=3, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
    elif op_id == 15:
        br_op = layers.AveragePooling2D(pool_size=5, strides=x_stride, padding='same')
        x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]

    return br_op, x_shape, x_id_fact_reduce


class Node(keras.Model):
    def __init__(self, x_id, x_op_id, y_id, y_op_id, x_shape, y_shape, channels, strides=1, drop_path_keep_prob=None,
                 cell_id=0, num_cells=0, steps=0):
        super(Node, self).__init__()
        self.channels = channels
        self.strides = strides
        self.drop_path_keep_prob = drop_path_keep_prob
        self.cell_id = cell_id
        self.num_cells = num_cells
        self.steps = steps
        self.x_id = x_id
        self.x_op_id = x_op_id
        self.y_id = y_id
        self.y_op_id = y_op_id

        self.x_op, self.x_shape, self.x_id_fact_reduce = op(self.x_id, self.x_op_id, list(x_shape), self.channels, self.strides)
        self.y_op, self.y_shape, self.y_id_fact_reduce = op(self.y_id, self.y_op_id, list(y_shape), self.channels, self.strides)

        assert self.x_shape[0] == self.y_shape[0] and self.x_shape[1] == self.y_shape[1]
        self.out_shape = list(self.x_shape)

    def call(self, x, y=None, step=None, training=False, **kwargs):
        x = self.x_op(x, training=training)
        if self.x_id_fact_reduce is not None:
            x = self.x_id_fact_reduce(x, training=training)
        if self.drop_path_keep_prob is not None and training:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.cell_id, self.num_cells, step, self.steps)

        y = self.y_op(y)
        if self.y_id_fact_reduce is not None:
            y = self.y_id_fact_reduce(y, training=training)
        if self.drop_path_keep_prob is not None and training:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.cell_id, self.num_cells, step, self.steps)

        out = x + y
        return out


class Cell(keras.Model):
    # A cell is a convolutional neural network containing B nodes.
    def __init__(self, arch, in_shapes, channels, is_reduction, cell_id, num_cells, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        assert len(in_shapes) == 2
        self.arch = arch
        self.is_reduction = is_reduction
        self.cell_id = cell_id
        self.num_cells = num_cells
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.num_nodes = len(arch) // 4
        self.used = [0] * (self.num_nodes + 2)

        # maybe calibrate size
        out_shapes = [list(in_shapes[0]), list(in_shapes[1])]
        self.calibrate = CalibrateSize(out_shapes, channels)
        out_shapes = self.calibrate.out_shapes

        self.nodes = []
        strides = 2 if self.is_reduction else 1
        for i in range(self.num_nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            x_shape, y_shape = out_shapes[x_id], out_shapes[y_id]
            node = Node(x_id, x_op, y_id, y_op, x_shape, y_shape, channels, strides, self.drop_path_keep_prob, cell_id,
                        num_cells, steps)
            self.nodes.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1
            out_shapes.append(node.out_shape)

        self.concat_index = [i for i in range(self.num_nodes + 2) if self.used[i] == 0]
        out_hw = min([shape[0] for i, shape in enumerate(out_shapes) if i in self.concat_index])
        self.final_combine = FinalCombine(out_shapes, out_hw, channels, self.concat_index)
        self.out_shape = [out_hw, out_hw, channels * len(self.concat_index)]

    def call(self, s0, s1=None, step=None, training=False, **kwargs):
        s0, s1 = self.calibrate(s0, s1, training=training)
        states = [s0, s1]
        for i in range(self.num_nodes):
            x_id = self.arch[4 * i]
            y_id = self.arch[4 * i + 2]
            x = states[x_id]
            y = states[y_id]
            out = self.nodes[i](x, y, step, training=training)
            states.append(out)
        return self.final_combine(states, training=training)


class NASNetworkCIFAR(keras.Model):
    def __init__(self,
                 classes,
                 reduce_distance,
                 num_nodes,
                 channels,
                 keep_prob,
                 drop_path_keep_prob,
                 use_aux_head,
                 steps,
                 arch):
        super(NASNetworkCIFAR, self).__init__()
        self.classes = classes
        self.reduce_distance = reduce_distance
        self.num_nodes = num_nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps

        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list) and len(arch) == 2:
            arch = arch[0] + arch[1]

        self.normal_arch = arch[:4 * self.num_nodes]
        self.reduce_arch = arch[4 * self.num_nodes:]

        # why two reduce cells here ??? why not three???
        # 0-5, 6, 7-12, 13, 14-19, 20 => [6, 13, 20]
        # i found 3 reduce cells is not good, so try back to 2 reduce cells as paper's work
        self.reduce_cells_idx = [self.reduce_distance, 2*self.reduce_distance+1, 3*reduce_distance+2]
        self.num_cells = self.reduce_distance * 3 + 2 # (6 normal cells * 3 + 2 reduce cells)

        if self.use_aux_head:
            self.aux_head_index = self.reduce_cells_idx[-1]  # do auxiliary head after the last pooling cell

        # stem_multiplier = 3 # why need this? why need stem layer? why multiply channels and not split.
        # self.channels = self.channels * stem_multiplier
        self.stem = keras.Sequential(
            [layers.Conv2D(filters=self.channels, kernel_size=3, padding='same', use_bias=False),
            # in_channels=3, out_channels=self.channels
            layers.BatchNormalization(axis=-1)]
        )

        out_shapes = [[32, 32, self.channels], [32, 32, self.channels]] # after stem layer
        self.cells = []
        for i in range(self.num_cells):
            if i in self.reduce_cells_idx:
                self.channels *= 2
                cell = Cell(self.reduce_arch, out_shapes, self.channels, True, i, self.num_cells, self.steps,
                            self.drop_path_keep_prob)
            else:
                cell = Cell(self.normal_arch, out_shapes, self.channels, False, i, self.num_cells, self.steps,
                            self.drop_path_keep_prob)
            self.cells.append(cell)
            out_shapes = [out_shapes[-1], cell.out_shape]

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(out_shapes[-1][-1], classes)

        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_pooling = layers.AveragePooling2D(pool_size=(out_shapes[-1][-3], out_shapes[-1][-2]))
        self.dropout = layers.Dropout(1 - self.keep_prob)
        self.classifier = layers.Dense(classes) # out_shapes[-1][-1], num_channels to classess

        # self.init_parameters()

    def call(self, input, step=None, training=False, **kwargs):
        aux_logits = None
        s0 = s1 = self.stem(input, training=training)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step, training=training)
            if self.use_aux_head and i == self.aux_head_index and training:
                aux_logits = self.auxiliary_head(s1, training=training)
        out = s1
        out = self.global_pooling(out)
        if training:
            out = self.dropout(out)
        out = tf.reshape(out, [out.shape[0], -1])
        logits = self.classifier(out)
        return logits, aux_logits


# class NASNetworkImageNet(nn.Module):
#     def __init__(self, args, classes, num_cells, num_nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps,
#                  arch):
#         super(NASNetworkImageNet, self).__init__()
#         self.args = args
#         self.classes = classes
#         self.num_cells = num_cells
#         self.num_nodes = num_nodes
#         self.channels = channels
#         self.keep_prob = keep_prob
#         self.drop_path_keep_prob = drop_path_keep_prob
#         self.use_aux_head = use_aux_head
#         self.steps = steps
#         arch = list(map(int, arch.strip().split()))
#         self.normal_arch = arch[:4 * self.num_nodes]
#         self.reduce_arch = arch[4 * self.num_nodes:]
#
#         self.pool_cells = [self.num_cells, 2 * self.num_cells + 1]
#         self.num_cells = self.num_cells * 3
#
#         if self.use_aux_head:
#             self.aux_head_index = self.pool_cells[-1]
#
#         channels = self.channels
#         self.stem0 = nn.Sequential(
#             nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#         )
#         self.stem1 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#         )
#         outs = [[56, 56, channels], [28, 28, channels]]
#         channels = self.channels
#         self.cells = nn.ModuleList()
#         for i in range(self.num_cells + 2):
#             if i not in self.pool_cells:
#                 cell = Cell(self.normal_arch, outs, channels, False, i, self.num_cells + 2, self.steps,
#                             self.drop_path_keep_prob)
#             else:
#                 channels *= 2
#                 cell = Cell(self.reduce_arch, outs, channels, True, i, self.num_cells + 2, self.steps,
#                             self.drop_path_keep_prob)
#             self.cells.append(cell)
#             outs = [outs[-1], cell.out_shape]
#
#             if self.use_aux_head and i == self.aux_head_index:
#                 self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)
#
#         self.global_pooling = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(1 - self.keep_prob)
#         self.classifier = nn.Linear(outs[-1][-1], classes)
#
#         self.init_parameters()
#
#     def init_parameters(self):
#         for w in self.parameters():
#             if w.data.dim() >= 2:
#                 nn.init.kaiming_normal_(w.data)
#
#     def call(self, input, step=None, **kwargs):
#         aux_logits = None
#         s0 = self.stem0(input)
#         s1 = self.stem1(s0)
#         for i, cell in enumerate(self.cells):
#             s0, s1 = s1, cell(s0, s1, step)
#             if self.use_aux_head and i == self.aux_head_index and self.training:
#                 aux_logits = self.auxiliary_head(s1)
#
#         out = s1
#         out = self.global_pooling(out)
#         out = self.dropout(out)
#         logits = self.classifier(out.view(out.size(0), -1))
#         return logits, aux_logits

