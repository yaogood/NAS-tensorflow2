import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class Decoder(keras.Model):
    def __init__(self,
                 search_space,
                 embedding_size,
                 hidden_size,
                 dropout,
                 ):
        super(Decoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocal_size = self.ss.vocal_size + 1 # 0-th is for the initial input
        self.dropout = dropout
        self.ss = search_space

        self.input_embedding = layers.Embedding(self.vocal_size, self.embedding_size) # for auto-aggressive
        self.dropout = layers.Dropout(self.dropout)

        self.lstm_cell = layers.LSTMCell(self.hidden_size, dropout=self.dropout)

        self.output_classifier = []
        for i in range(self.ss.count * self.ss.num_layers): # different kinds of tokens
            self.decoders.append(keras.layers.Dense(units=self.ss[i]['size']))

    def attention(self, current_hidden, encoder_outputs):
        pass

    def call(self, input_tokens, encoder_hidden=None, encoder_outputs=None, training=None, **kwargs): # inputs shape: [bsize, features]
        batch_size = encoder_hidden.shape[0]
        rnn_input = tf.zeros([batch_size, self.embedding_size + self.hidden_size])

        all_logits = []
        all_pred_tokens = []

        for i in range(self.ss.count * self.ss.num_layers):
            rnn_out, hidden = self.lstm_cell(rnn_input, encoder_hidden, training=training)
            context_vec = self.attention(hidden, encoder_outputs)
            out = tf.concat(rnn_out, context_vec)

            logits = self.decoders[i](out) # logits shape: [batch_size, len(current_token)]
            all_logits.append(logits)

            logits_prob = tf.nn.softmax(logits)  # probs shape: [batch_size, len(current_token)]
            # log_probs = tf.nn.log_softmax(logits)
            pred_token = tf.argmax(logits_prob, axis=-1)
            pred_token = tf.expand_dims(pred_token, -1)
            all_pred_tokens.append(pred_token)

            embed = self.input_embedding(pred_token)
            if training:
                embed = self.dropout(embed)
            rnn_input = tf.concat(embed, context_vec, axis=-1)


        return all_logits, all_pred_tokens






