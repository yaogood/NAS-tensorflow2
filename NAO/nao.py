import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class Encoder(keras.Model):
    # 1 layer LSTM + 3 layers MLP + regressor
    def __init__(self,
                 lstm_num_layers,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 lstm_dropout,
                 mlp_num_layers,
                 mlp_hidden_size,
                 mlp_dropout):
        super(Encoder, self).__init__()

        self.lstm_num_layers = lstm_num_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout

        self.embedding_layer = layers.Embedding(self.vocab_size, self.embedding_size)
        self.dropout_layer = layers.Dropout(lstm_dropout)

        self.lstm = []
        for _ in range(self.lstm_num_layers):
            self.lstm.append(layers.LSTM(self.hidden_size, return_sequences=True, dropout=lstm_dropout))

        self.mlp = keras.Sequential()
        for _ in range(self.mlp_num_layers):
            self.mlp.add(layers.Dense(self.mlp_hidden_size))
            self.mlp.add(layers.Dropout(self.mlp_dropout))

        self.regressor = layers.Dense(1)

    def call(self, input_tokens, training=None, **kwargs): # x shape: [batch_size, sentence_lenth, vocal_lenth]
        x = self.embedding_layer(input_tokens)
        if training:
            x = self.dropout_layer(x)

        hidden = None
        for i in range(self.lstm_num_layers):
            outs, hidden = self.lstm[i](x, training=training)
            x = outs

        encoder_outputs = keras.utils.normalize(x, axis=-1, order=2)
        # encoder_outputs shape: [batch_size, seq_lenth, hidden_size]
        # encoder_hidden shape: [hs:[batch_size, hidden_size], cs:[batch_size, hidden_size]]

        arch_embedding = tf.math.reduce_mean(encoder_outputs, axis=1)
        arch_embedding = keras.utils.normalize(arch_embedding, axis=-1, order=2)
        # arch_embedding shape: [batch_size, hidden_size]

        logits = self.mlp(arch_embedding, training=training)
        logits = self.regressor(logits)
        #logits: [bsize, 1]
        prediction = tf.math.sigmoid(logits) # predicted accuracy of selected architectures
        # prediction: [bsize, 1]

        assert hidden is not None
        return encoder_outputs, hidden, arch_embedding, prediction

    def infer(self, input_tokens, lr_lambda, direction=1):
        with tf.GradientTape() as tape:
            encoder_outputs, encoder_hidden, arch_embedding, prediction = self(input_tokens)

        gradients = tape.gradient(prediction, encoder_outputs)
        new_encoder_outputs = encoder_outputs + direction * lr_lambda * gradients
        new_encoder_outputs = keras.utils.normalize(new_encoder_outputs, axis=-1, order=2)

        new_arch_embedding = tf.math.reduce_mean(new_encoder_outputs, axis=1)
        new_arch_embedding = keras.utils.normalize(new_arch_embedding, axis=-1, order=2)

        return encoder_outputs, encoder_hidden, arch_embedding, prediction, new_encoder_outputs, new_arch_embedding


class Decoder(keras.Model):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self,
                 lstm_num_layers,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 dropout
                 ):
        super(Decoder, self).__init__()

        self.lstm_num_layers = lstm_num_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding_layer = layers.Embedding(self.vocab_size, self.embedding_size) # for auto-aggressive
        self.dropout_layer = layers.Dropout(dropout)

        self.lstm_cell = []
        for _ in range(self.lstm_num_layers):
            self.lstm_cell.append(layers.LSTMCell(self.hidden_size, dropout=self.dropout))

        self.output_classifier = []
        for i in range(self.ss.count): # different kinds of tokens
            self.output_classifier.append(keras.layers.Dense(units=self.ss[i]['size']))

    def attention(self, current_hidden, encoder_outputs):
        pass

    def call(self, input_tokens, encoder_hidden=None, encoder_outputs=None, training=None, **kwargs):
        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()

        if input_tokens is None:
            inference = True
        else:
            inference = False

        x, batch_size, length = self._validate_args(input_tokens, encoder_hidden, encoder_outputs)

        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

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

    def _validate_args(self, input_tokens, hidden, encoder_outputs):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if input_tokens is None and hidden is None:
            batch_size = 1
        else:
            if input_tokens is not None:
                batch_size = input_tokens.shape[0]
            else:
                batch_size = hidden[0].shape[1]

        # inference, set default input and max decoding length
        if input_tokens is None:
            input_tokens = tf.Variable([self.sos_id] * batch_size, dtype=tf.int32).reshape(batch_size, 1)
            max_length = self.length
        else:
            max_length =input_tokens.shape[1]

        return input_tokens, batch_size, max_length


class NAO(keras.Model):
    def __init__(self,
                 # search_space,
                 encoder_vocab_size,
                 encoder_lstm_num_layers,
                 encoder_embedding_size,
                 encoder_hidden_size,
                 encoder_dropout,
                 mlp_num_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 decoder_lstm_num_layers,
                 decoder_vocab_size,
                 decoder_embedding_size,
                 decoder_hidden_size,
                 decoder_dropout
                 ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            # search_space,
            encoder_lstm_num_layers,
            encoder_vocab_size,
            encoder_embedding_size,
            encoder_hidden_size,
            encoder_dropout,
            mlp_num_layers,
            mlp_hidden_size,
            mlp_dropout,
        )
        self.decoder = Decoder(
            # search_space,
            decoder_lstm_num_layers,
            decoder_vocab_size,
            decoder_embedding_size,
            decoder_hidden_size,
            decoder_dropout
        )

    def call(self, encoder_input_tokens, decoder_input_tokens=None, **kwargs):
        encoder_outputs, encoder_hidden, arch_embedding, pred_acc = self.encoder(encoder_input_tokens, training=True)
        decoder_hidden_initial = (arch_embedding.unsqueeze(0), arch_embedding.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(decoder_input_tokens, decoder_hidden_initial, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return pred_acc, decoder_outputs, arch

        all_logits, all_pred_tokens, decoder_hidden = self.decoder(decoder_input_tokens, encoder_hidden, encoder_outputs)
        return pred_acc, all_logits, all_pred_tokens

    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        all_logits, all_pred_tokens, decoder_hidden = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        new_arch = all_pred_tokens
        return new_arch





