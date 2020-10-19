import os
import  datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
if not os.path.exists('logs/'):
    os.makedirs('logs/')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

tf.random.set_seed(9)
np.random.seed(9)

class RNNPolicyNetwork(keras.Model):
    def __init__(self, units, search_space, batch_size=1, embedding_dim=20):
        super(RNNPolicyNetwork, self).__init__()
        tf.random.set_seed(9)
        np.random.seed(9)
        self.units = units
        self.ss = search_space
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.input_embedding = tf.Variable(
            tf.random.uniform([self.batch_size, self.embedding_dim], dtype=tf.dtypes.float32), trainable=True
        )
        self.encoders = []
        self.decoders = []
        for i in range(self.ss.count):
            size = self.ss[i]['size']
            self.encoders.append(keras.layers.Embedding(size, self.embedding_dim, embeddings_initializer='uniform'))
        for i in range(self.ss.count * self.ss.num_layers):
            size = self.ss[i]['size']
            self.decoders.append(keras.layers.Dense(units=size))
        # Use LSTM as the controller here
        self.nas_cell = keras.layers.LSTMCell(self.units, dropout=0.1)

    # def get_config(self):
    #     config = super(RNNPolicyNetwork, self).get_config()
    #     config.update({'units': self.units})
    #     return config

    def call(self, inputs, training=None, prev_hidden=None, with_trainable_input=False):
        # inputs: [batch_size, embedding_dim]
        if with_trainable_input:
            cell_input = self.input_embedding
        else:
            cell_input = tf.zeros([self.batch_size, self.embedding_dim])

        if prev_hidden:
            hidden = prev_hidden
        else:
            hidden = [tf.zeros([self.batch_size, self.units]), tf.zeros([self.batch_size, self.units])]

        entropies = []
        actions = []
        selected_log_probs = []
        logits_buffer = []
        # print("logits probs:")
        # a flat list of chained input-output to the RNN
        for i in range(self.ss.count * self.ss.num_layers):
            # cell_input.shape: [batch_size, embedding_dim]
            # hidden[0].shape: [batch_size, hidden_units],  hidden[1].shape: [batch_size, hidden_units]
            out, hidden = self.nas_cell(cell_input, hidden, training=training)
            # logits shape: [batch_size, len(current_token)]
            logits = self.decoders[i](out)
            # probs shape: [batch_size, len(current_token)]
            probs = tf.nn.softmax(logits)
            print(probs.numpy())
            log_probs = tf.nn.log_softmax(logits)
            # entropy shape: [batch_size, 1]
            entropies.append(tf.reduce_sum(-(log_probs * probs), axis=1, keepdims=True))
            # action shape: [batch_size, 1]
            if not training:
                action = tf.argmax(probs, axis=-1)
                action = tf.expand_dims(action, -1)
                actions.append(action)
            else:
                action = tf.random.categorical(logits, 1)
                actions.append(action)
                # selected_log_prob shape: [batch_size, 1]
                selected_log_prob = tf.gather(log_probs, action, batch_dims=1)  # batch_size * 1
                selected_log_probs.append(selected_log_prob)
            # embedding lookup of this state using its state weights ; reuse weights
            embed_idx = tf.cast(action, tf.int32)
            # cell_input = tf.nn.embedding_lookup(self.embedding_weights[i % self.ss.count], embed_idx)
            cell_input = self.encoders[i % self.ss.count ](embed_idx)
            cell_input = tf.squeeze(cell_input, axis=1)

        # archs shape: [batch_size, token_string_length]
        archs = tf.concat(actions, axis=-1)
        if not training:
            return archs

        # archs shape: [batch_size, token_string_length]
        entropies = tf.concat(entropies, axis=-1)
        # selected_log_probs shape: [batch_size, token_string_length]
        selected_log_probs = tf.concat(selected_log_probs, axis=-1)

        return archs, selected_log_probs, entropies, hidden


def train_controller(model, search_space, env, optimizer, steps,
                     reg_strength=0.001,
                     clip_norm=15.0,
                     entropy_coeff = 0.001,
                     ema_alpha=0.15,
                     hidden=None,
                     more_exploration=False,
                     with_regularation=False):


    arch_buffer = []
    accuracy_buffer = []
    ema_reward_buffer = []
    loss_buffer = []

    ema_baseline = None

    for step in range(steps):
        print('\n')
        print('-' * 40,"step %d:" % (step+1), '-' * 40)

        with tf.GradientTape() as tape:
            archs, selected_log_probs, entropies, hidden = \
                model(None, training=True, prev_hidden=hidden, with_trainable_input=False)

            current_batch_acc = []
            acc_buffer = []

            with tape.stop_recording():
                for arch in archs:
                    actions = search_space.to_values(arch.numpy())
                    acc = env.get_rewards(actions)
                    # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Training: ", actions, " --> ", acc)
                    # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
                    current_batch_acc.append(tf.constant(acc, dtype=tf.dtypes.float32))
                    acc_buffer.append(acc)
                # current_batch_acc shape: [batch_size] => [batch_size, 1]
                current_batch_acc = tf.expand_dims(current_batch_acc, -1)

            if more_exploration:
                current_batch_rewards = current_batch_acc + entropy_coeff * entropies
                entropy_coeff = entropy_coeff * (1 - step/steps)
            else:
                current_batch_rewards = current_batch_acc
            # policy_loss shape: [batch_sz, ts_length] * [batch_sz, 1] => [batch_sz, ts_length] => ()


            with tape.stop_recording():
                if ema_baseline is None:
                    ema_baseline = current_batch_rewards
                else:
                    ema_baseline = ema_alpha * current_batch_rewards + (1 - ema_alpha) * ema_baseline

            policy_loss = tf.reduce_sum((-1) * selected_log_probs * (current_batch_rewards - ema_baseline))

            reg_loss = reg_strength * tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in model.trainable_variables]))
            if with_regularation:
                policy_loss = policy_loss + reg_loss


        print("policy loss: ", policy_loss.numpy())
        gradients = tape.gradient(policy_loss, model.trainable_variables)
        # clip and apply gradient
        if clip_norm != 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        arch_buffer.append(archs.numpy())
        accuracy_buffer.append(acc_buffer)
        ema_reward_buffer.append(ema_baseline.numpy())
        loss_buffer.append(policy_loss.numpy())

    return arch_buffer, accuracy_buffer, ema_reward_buffer, loss_buffer, hidden
