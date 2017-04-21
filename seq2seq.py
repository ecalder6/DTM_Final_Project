import tensorflow as tf

class Seq2Seq(object):
    '''
    In-house seq2seq
    '''

    def __init__(self, tweets, replies, hidden_size, layers, \
            batch_size, emb_size, vocab_size, seq_max_len):

        self._batch_size = batch_size
        self._emb_size = emb_size
        self._max_seq_len = seq_max_len

        # Word embeddings
        self._word_embeddings = tf.get_variable("embedding", [vocab_size, emb_size])
        emb_tweet = tf.nn.embedding_lookup(self._word_embeddings, tweets)

        # Encoder
        inner_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        enc_cell = tf.contrib.rnn.MultiRNNCell([inner_cell] * layers)
        _, self.thought_vector = tf.nn.dynamic_rnn(
            enc_cell, emb_tweet, swap_memory=True, dtype=tf.float32)

        # decoder
        reply_input = tf.concat(  # Add GO token to start
            [tf.zeros(shape=(batch_size, 1), dtype=tf.int64), replies[:, :seq_max_len-1]], axis=1)
        emb_reply_input = tf.nn.embedding_lookup(self._word_embeddings, reply_input)
        self.dec_cell = tf.contrib.rnn.OutputProjectionWrapper(enc_cell, vocab_size)
        with tf.variable_scope("decoder"):
            self.dec_out, _ = tf.nn.dynamic_rnn(self.dec_cell, emb_reply_input,  \
                                initial_state=self.thought_vector, \
                                swap_memory=True, dtype=tf.float32)



    def get_loss(self, replies):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dec_out, labels=replies)
        loss = tf.reduce_sum(xent, axis=[1])
        avg_loss = tf.reduce_mean(loss)
        return avg_loss

    def sample(self, sample_temp):
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:  # time == 0
                next_cell_state = self.thought_vector  # state from the encoder
                next_input = tf.zeros([self._batch_size], dtype=tf.int64)  # GO symbol
                next_input = tf.nn.embedding_lookup(self._word_embeddings, next_input)
                emit_output = tf.zeros([], dtype=tf.int64)
            else:
                next_cell_state = cell_state
                sample = tf.squeeze(tf.multinomial(cell_output / sample_temp, 1))
                print(sample)
                emb_sample = tf.nn.embedding_lookup(self._word_embeddings, sample)
                next_input = emb_sample
                emit_output = sample
            elements_finished = time >= tf.constant(self._max_seq_len, shape=(self._batch_size,))
            finished = tf.reduce_all(elements_finished)
            print(next_input)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([self._batch_size, self._emb_size], dtype=tf.float32),
                lambda: next_input)
            print(next_input)
            next_loop_state = None
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        with tf.variable_scope("decoder", reuse=True):
            outputs_ta, _, _ = tf.nn.raw_rnn(self.dec_cell, loop_fn, swap_memory=True)
            sample = outputs_ta.stack()
            return sample

    def train(self, loss, lr):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1.0)
        optimizer = tf.train.RMSPropOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op

