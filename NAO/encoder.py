import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class Encoder(keras.Model):
    # 1 layer LSTM + 3 layers MLP + regressor
    def __init__(self,
                 search_space,
                 embedding_size,
                 hidden_size,
                 dropout,
                 mlp_num_layers,
                 mlp_hidden_size,
                 mlp_dropout):
        super(Encoder, self).__init__()

        self.ss = search_space
        self.vocal_size = self.ss.vocal_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout

        self.input_embedding = layers.Embedding(self.vocal_size, self.embedding_size)
        self.dropout = layers.Dropout(dropout)

        self.lstm= layers.LSTM(self.hidden_size, return_sequences=True, dropout=dropout)

        self.mlp = keras.Sequential()
        for i in range(self.mlp_num_layers):
            self.mlp.add(layers.Dense(self.mlp_hidden_size))
            self.mlp.add(layers.Dropout(self.mlp_dropout))

        self.regressor = layers.Dense(1)

    def call(self, input_tokens, training=None, **kwargs): # x shape: [batch_size, sentence_lenth, vocal_lenth]
        embed = self.embedding(input_tokens)
        if training:
            embed = self.dropout(embed)

        encoder_outputs, encoder_hidden = self.lstm(embed, training=training)
        encoder_outputs = keras.utils.normalize(encoder_outputs, axis=-1, order=2)
        # encoder_outputs shape: [batch_size, seq_lenth, hidden_size]
        # encoder_hidden shape: [hs:[batch_size, hidden_size], cs:[batch_size, hidden_size]]

        arch_embedding = tf.math.reduce_mean(encoder_outputs, dim=1)
        arch_embedding = keras.utils.normalize(arch_embedding, axis=-1, order=2)
        # arch_embedding shape: [batch_size, hidden_size]

        logits = self.mlp(arch_embedding, training=training)
        logits = self.regressor(logits)
        #logits: [bsize, 1]

        prediction = tf.math.sigmoid(logits) # predicted accuracy of selected architectures
        # prediction: [bsize, 1]

        return encoder_outputs, encoder_hidden, arch_embedding, prediction

    def infer(self, inputs, predict_lambda, direction=-1):
        with tf.GradientTape() as tape:
            encoder_outputs, encoder_hidden, arch_embedding, prediction = self(inputs)

        gradients = tape.gradient(prediction, encoder_outputs)
        new_encoder_outputs = encoder_outputs + direction * predict_lambda * gradients
        new_encoder_outputs = keras.utils.normalize(new_encoder_outputs, axis=-1, order=2)

        new_arch_embedding = tf.math.reduce_mean(new_encoder_outputs, dim=1)
        new_arch_embedding = keras.utils.normalize(new_arch_embedding, axis=-1, order=2)

        return encoder_outputs, encoder_hidden, arch_embedding, prediction, new_encoder_outputs, new_arch_embedding