import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

class LSTMVAE(object):
    '''
    In-house seq2seq
    '''

    def __init__(self, x, y, batch_size, emb_size, \
        latent_size, vocab_size, seq_max_len):
        '''
        Initializes a LSTM-VAE

        Arguments:
            x (tensor): A tensor holding the input to our network.
            y (tensor): A tensor holding the output to our network
            layers (int):   The number of layers we use in the encoder
            batch_size (int):   The size of our batches
            emb_size (int): word embedding dimension also the length of the
                            state being passed around inside a LSTM cell
            latent_size (int):  The number of latent variables in our VAE
            vocab_size (int): length of vocabulary
            seq_max_len (int):  limit on length of input and output
        '''

        self._batch_size = batch_size
        self._emb_size = emb_size
        self._max_len = seq_max_len

        ### Encoder ###
        # emb_size is the size of the internal state in each cell.
        # Note that cell state and hidden/output state are same size
        enc_cell = tf.contrib.rnn.BasicLSTMCell(emb_size)

        # Word embeddings
        # Gives a embedding for each word in the vocab
        self._word_embeddings = tf.get_variable("embedding", [vocab_size, emb_size])

        # emb_tweet is a tensor
        # if input (x) is n by d then emb_input is
        # n by d by emb_size
        emb_input = tf.nn.embedding_lookup(self._word_embeddings, x)

        # dynamic_rnn creates RNN and creates an output, state tensor
        # since this is the encoder we just care about the output state
        # m_state and c_state are tuples of length (layers). Each item in the
        # tuple is a state tuple. Each state tuple is size
        # batch_size by embed size.
        # NOTE: shape=(?,emb_size). ? means None or that it is not known.
        # At run time this gets filled as the batch size
        _, c_state = tf.nn.dynamic_rnn( \
                    enc_cell, emb_input, swap_memory=True, dtype=tf.float32)


        ### VAE ###
        # Set up the audoencoder weights and layers
        # Why get_variable vs. variable here?
        w1 = tf.get_variable("w1", shape=[emb_size, latent_size],
             initializer=tf.contrib.layers.xavier_initializer())
        log_sigma_w1 = tf.get_variable("log_sigma_w1", shape=[emb_size, latent_size],
            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.zeros([latent_size], dtype=tf.float32))
        log_sigma_b1 = tf.Variable(tf.zeros([latent_size], dtype=tf.float32))
        w2 = tf.Variable(tf.zeros([latent_size, emb_size], dtype=tf.float32))
        b2 = tf.Variable(tf.zeros([emb_size], dtype=tf.float32))

        # Here we get the parameters of the latent posterior
        z_mean = tf.add(tf.matmul(c_state.c, w1), b1)
        z_log_sigma_sq = tf.add(tf.matmul(c_state.c, log_sigma_w1), log_sigma_b1)
        # Sample random noise from gaussian
        eps = tf.random_normal(tf.stack([tf.shape(c_state.c)[0], latent_size]), 0, 1, dtype = tf.float32)

        # Get z
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
        # Kind of "decode" z into something for the decoder
        reconstruction = tf.add(tf.matmul(z, w2), b2)

        # State after VAE
        self._latent_state = LSTMStateTuple(reconstruction, c_state.h)

        ### Decoder ###
        reply_input = tf.concat(  # Add GO token to start
            [tf.zeros(shape=(batch_size, 1), dtype=tf.int64), y[:, :seq_max_len-1]], axis=1)
        emb_reply_input = tf.nn.embedding_lookup(self._word_embeddings, reply_input)
        self.dec_cell = tf.contrib.rnn.OutputProjectionWrapper(enc_cell, vocab_size)
        with tf.variable_scope("decoder"):
            self.dec_out, _ = tf.nn.dynamic_rnn(self.dec_cell, emb_reply_input,  \
                                initial_state=self._latent_state, \
                                swap_memory=True, dtype=tf.float32)

        ### Cost ###
        # Regular LSTM loss
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dec_out, labels=y)
        dec_loss = tf.reduce_sum(xent, axis=[1])
        # Latent loss
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                           - tf.square(z_mean)
                                           - tf.exp(z_log_sigma_sq), 1)
        self.loss = tf.reduce_mean(latent_loss + dec_loss)


    def sample(self, sample_temp):
        '''
        Returns a sample from the decoder. loop_fn is what's ran
        at each cell from the start of the decoder to the end.
        outputs_ta is the output of the final cell.
        '''
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:  # time == 0
                next_cell_state = self._latent_state  # state from the encoder
                next_input = tf.zeros([self._batch_size], dtype=tf.int64)  # GO symbol
                next_input = tf.nn.embedding_lookup(self._word_embeddings, next_input)
                emit_output = tf.zeros([], dtype=tf.int64)
            else:
                next_cell_state = cell_state
                sample = tf.squeeze(tf.multinomial(cell_output / sample_temp, 1))
                emb_sample = tf.nn.embedding_lookup(self._word_embeddings, sample)
                next_input = emb_sample
                emit_output = sample
            elements_finished = time >= tf.constant(self._max_len, shape=(self._batch_size,))
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([self._batch_size, self._emb_size], dtype=tf.float32),
                lambda: next_input)
            next_loop_state = None
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        with tf.variable_scope("decoder", reuse=True):
            outputs_ta, _, _ = tf.nn.raw_rnn(self.dec_cell, loop_fn, swap_memory=True)
            sample = outputs_ta.stack()
            return sample

    def train(self, lr):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1.0)
        optimizer = tf.train.RMSPropOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op
