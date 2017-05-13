import tensorflow as tf
import math
import time
from tensorflow.contrib.rnn import LSTMStateTuple


def to_eng(ids, ix_to_word):
    output = ""
    for id in ids:
        if id:
            output += ix_to_word[id] + " "
    return output

class RNNLMVAE(object):
    '''
    Single layer recurrent neural network language model with Variational autoencoder 
    architecture, based off https://arxiv.org/pdf/1511.06349.pdf .
    Uses TensorFlow 1.0.

    Args:
        x (tensor):         A tensor holding the input to our network.
        layers (int):       The number of layers we use in the encoder
        batch_size (int):   The size of our batches
        emb_size (int):     word embedding dimension also the length of the
                            state being passed around inside a LSTM cell
        latent_size (int):  The number of latent variables in our VAE
        vocab_size (int):   length of vocabulary
        seq_max_len (int):  limit on length of input and output

    '''
    def __init__(   self, 
                    batch_size, 
                    emb_size,
                    latent_size, 
                    vocab_size, 
                    seq_max_len,
                    mutual_lambda = 0.1):

        self.batch_size = batch_size
        self.emb_size = emb_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.max_len = seq_max_len
        self.mutual_lambda = mutual_lambda
        self.kl_anneal = tf.Variable(1.0)

        # Word embedding holder
        self.word_embeddings = tf.get_variable("embedding", [vocab_size, emb_size])

    def train(self,
            x,
            y, 
            iterations,
            reader_meta,
            use_mutual=True,
            mutual_lambda=0.1, 
            optimizer=tf.train.GradientDescentOptimizer,
            learning_rate = 0.01,
            anneal=True):

        # TODO: Again we could've put this code into construct_graph but the dependency on
        # y makes it hard. 
        reply_input = tf.concat(  # Add GO token to start
            [tf.zeros(shape=(self.batch_size, 1), dtype=tf.int64), y[:, :self.max_len-1]], axis=1)
        emb_reply_input = tf.nn.embedding_lookup(self.word_embeddings, reply_input)

        with tf.variable_scope("decoder"):
            self.dec_out, _ = tf.nn.dynamic_rnn(self.dec_cell, emb_reply_input,  \
                                initial_state=self.latent_features, \
                                swap_memory=True, dtype=tf.float32)

        updates = self.get_updates(self.dec_out, y, optimizer, learning_rate)
        train_start_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            losses = [[] for _ in updates]
            sample_outputs = []
            
            for step in range(iterations):
                iteration_start_time = time.time()
                new_anneal = self.kl_anneal.assign(1/(1+(1.01)**(-step+999)))
                # TODO: Threshold variance
                threshold_var = tf.maximum(self.z_log_cov_sq, 0.25)
                sess.run([new_anneal, threshold_var])
                i = 0
                for up, loss in updates:
                    _, l = sess.run([up,loss])
                    losses[i].append(l)
                    i += 1
                if step % 100 == 0 or True:
                    actual_x, actual_y, sampled_y, z = sess.run([x, y, self.sample(), self.z])
                    sample_outputs.append((actual_x, actual_y, sampled_y))

                    print("\nIteration: ", step+1)
                    print("Duration for this epoch: ", time.time()-iteration_start_time )
                    print("Objective loss: %.3f" % losses[0][-1])
                    print("KL loss: %.3f" % losses[1][-1])
                    if use_mutual:
                        print("Mutual loss: %.3f\n" % losses[2][-1])
                    
                    for i in range(5):
                        print("====================================================")
                        # Windows: chcp 65001
                        print(to_eng(actual_x[i], reader_meta['idx2w']), "-->", to_eng(sampled_y[:, i], reader_meta['idx2w']))
                        print("True reply: ", to_eng(actual_y[i], reader_meta['idx2w']))
                        print("First 5 z values: ", z[i][:5])
                        print("====================================================\n\n")
                        # print(c[i])
                        # print(r[:,i])
            print("Total training time: ", time.time() - train_start_time)

    def get_updates(self, dec_out, y, optimizer, learning_rate, use_mutual=True):
        # Get the normal model's losses
        dec_loss, kl_loss = self.get_vae_losses(dec_out, y)

        # different sets of variables to train
        all_tvar = tf.trainable_variables()
        vae_tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae')

        # Could try to have a different optimizer for each set of variables
        trainer = optimizer(learning_rate)
        
        all_grad = trainer.compute_gradients(dec_loss, all_tvar)
        vae_grad = trainer.compute_gradients(kl_loss, vae_tvar)

        update_all = trainer.apply_gradients(all_grad)
        update_vae = trainer.apply_gradients(vae_grad)
        updates = [(update_all, dec_loss), (update_vae, kl_loss)]

        if use_mutual:
            mutual_loss = self.get_mutual_loss()
            trainer_mutual = optimizer(learning_rate)
            mutual_grad = trainer.compute_gradients(mutual_loss, vae_tvar)
            update_mut = trainer.apply_gradients(mutual_grad)
            updates.append((update_mut, mutual_loss))
        return updates


    def get_vae_losses(self,dec_out, y):
        ### Cost ###
        # Decoder loss
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_out, labels=y)
        dec_loss = tf.reduce_sum(xent, axis=[1])
        
        # Latent loss
        # latent loss from the other dude's code
        kl_loss = -0.5 * self.kl_anneal * tf.reduce_sum(1 + self.z_log_cov_sq
                                        - tf.square(self.z_mean)
                                        - tf.exp(self.z_log_cov_sq), 1)
        return tf.reduce_mean(dec_loss), tf.reduce_mean(kl_loss)

    def get_mutual_loss(self):
        # Mutual information loss
        # Entropy of Q. There should be another expectation but we use the stochastic trick to take
        # one sample of x (the current x) and use that as an approximation (look at bottom of page 9
        # in VAE tutorial)
        return tf.reduce_mean(0.5 * tf.square( self.z - self.c ))  

    def construct_graph(self, x, use_vae=True, use_highway=True, forget_b=1.0):
        '''
        Construct the graph with highway layers and vae architecture.
        Encoder and decoder are basic Seq2Seq models. The decoder is fully
        made in train.

        TODO: It would be great to separate the construction of the graph from 
        seeing x but we use tf.records and queue runners and we don't know how to
        use placeholders with the queue runners. We would have to switch to feed
        dictionaries to release the dependency on x here.

        '''

        self.use_vae = use_vae
        self.use_highway = use_highway

        # emb_tweet is a tensor holding the word embeddings of the input
        # if input (x) is n by d then emb_input is
        # n by d by emb_size
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # I REALLY WANT TO GET RID OF THIS X.
        # TENSORFLOW STUPID DESIGN CHOICES.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        emb_input = tf.nn.embedding_lookup(self.word_embeddings, x)

        emb_size = self.emb_size
        latent_size = self.latent_size
        with tf.variable_scope("encoder"):
            ### Encoder ###
            # emb_size is the size of the internal state in each cell.
            # Note that cell state and hidden/output state are same size
            self.enc_cell = tf.contrib.rnn.BasicLSTMCell(emb_size, forget_bias=forget_b)

            # dynamic_rnn creates RNN and creates an output, state tensor
            # since this is the encoder we just care about the output state
            # m_state and c_state are tuples of length (layers). Each item in the
            # tuple is a state tuple. Each state tuple is size
            # batch_size by embed size.
            # NOTE: shape=(?,emb_size). ? means None or that it is not known.
            # At run time this gets filled as the batch size
            _, c_state = tf.nn.dynamic_rnn( \
                        self.enc_cell, emb_input, swap_memory=True, dtype=tf.float32)
            self.c = c_state

        ### VAE ###
        # Set up the audoencoder weights and layers
        # Why get_variable vs. variable here?
        if use_vae:
            with tf.variable_scope("vae"):
                w1 = tf.get_variable("mean_w1", shape=[emb_size, latent_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                log_cov_w1 = tf.get_variable("log_cov_w1", shape=[emb_size, latent_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.get_variable('mean_b1', initializer=tf.zeros([latent_size], dtype=tf.float32))
                log_cov_b1 = tf.get_variable('log_cov_b1', initializer=tf.zeros([latent_size], dtype=tf.float32))
                w2 = tf.get_variable('z_w2', initializer=tf.zeros([latent_size, emb_size], dtype=tf.float32))
                b2 = tf.get_variable('z_b2', initializer=tf.zeros([emb_size], dtype=tf.float32))

                # Here we get the parameters of the representation space
                z_mean = tf.add(tf.matmul(c_state.c, w1), b1)
                z_log_cov_sq = tf.add(tf.matmul(c_state.c, log_cov_w1), log_cov_b1)
                self.z_mean = z_mean
                self.z_log_cov_sq = z_log_cov_sq
                # Sample random noise from gaussian
                eps = tf.random_normal(tf.stack([tf.shape(c_state.c)[0], latent_size]), 0, 1, dtype = tf.float32)

                if use_highway:
                    ### Highway networks ###
                    with tf.variable_scope("highway"):
                        w1_t = tf.get_variable("mean_w1_t", shape=[emb_size, latent_size],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b1_t = tf.get_variable("mean_b1_t", initializer=tf.zeros([latent_size], dtype=tf.float32))
                        log_cov_w1_t = tf.get_variable("log_cov_w1_t", shape=[emb_size, latent_size],
                            initializer=tf.contrib.layers.xavier_initializer())
                        log_cov_b1_t = tf.get_variable("log_cov_b1_t", initializer=tf.zeros([latent_size], dtype=tf.float32))
                        w2_t = tf.get_variable("z_w2_t", initializer=tf.zeros([latent_size, emb_size], dtype=tf.float32))
                        b2_t = tf.get_variable("z_b2_t", initializer=tf.zeros([emb_size], dtype=tf.float32))
                        z_mean_t = tf.add(tf.matmul(c_state.c, w1_t), b1_t)
                        z_log_cov_sq_t = tf.add(tf.matmul(c_state.c, log_cov_w1_t), log_cov_b1_t)
                    self.z = tf.add(tf.multiply(z_mean, z_mean_t), tf.multiply(tf.sqrt(tf.exp(tf.multiply(z_log_cov_sq, z_log_cov_sq_t))), eps))
                    self.reconstruction = tf.multiply( tf.add(tf.matmul(self.z, w2_t), b2_t), tf.add(tf.matmul(self.z, w2), b2))
                else:
                    self.z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_cov_sq)), eps))
                    self.reconstruction = tf.add(tf.matmul(self.z, w2), b2)
            self.latent_features = LSTMStateTuple(self.reconstruction, c_state.h)
        else:
            self.latent_features = c_state

        ### Decoder ###
        with tf.variable_scope("decoder"):
            self.dec_cell = tf.contrib.rnn.OutputProjectionWrapper(self.enc_cell, self.vocab_size)

    def sample(self, sample_temp=0.7):
        '''
        Returns a sample from the decoder. loop_fn is what's ran
        at each cell from the start of the decoder to the end.
        outputs_ta is the output of the final cell.
        '''
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:  # time == 0
                next_cell_state = self.latent_features  # state from the encoder
                next_input = tf.zeros([self.batch_size], dtype=tf.int64)  # GO symbol
                next_input = tf.nn.embedding_lookup(self.word_embeddings, next_input)
                emit_output = tf.zeros([], dtype=tf.int64)
            else:
                next_cell_state = cell_state
                sample = tf.squeeze(tf.multinomial(cell_output / sample_temp, 1))
                emb_sample = tf.nn.embedding_lookup(self.word_embeddings, sample)
                next_input = emb_sample
                emit_output = sample
            elements_finished = time >= tf.constant(self.max_len, shape=(self.batch_size,))
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([self.batch_size, self.emb_size], dtype=tf.float32),
                lambda: next_input)
            next_loop_state = None
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        with tf.variable_scope("decoder", reuse=True):
            outputs_ta, _, _ = tf.nn.raw_rnn(self.dec_cell, loop_fn, swap_memory=True)
            sample = outputs_ta.stack()
            return sample