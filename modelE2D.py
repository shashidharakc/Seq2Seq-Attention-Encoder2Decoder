import numpy as np
import tensorflow as tf


class E2D(object):
    def __init__(self, constants_dictionary, data_dictionary):

        self.encoder_train_inp = data_dictionary['encoder_train_inp']
        self.source_embedding = data_dictionary['encoder_embedding']
        self.encoder_sequence_length = data_dictionary['encoder_sequence_length']
        encoder_emb_inp = tf.nn.embedding_lookup(self.source_embedding, self.encoder_train_inp)
        encoder_emb_inp_time_major = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])
        # encoder_sequence_length = tf.placeholder(tf.int32, shape=[None, ])

        self.decoder_train_inp = data_dictionary['decoder_train_inp']
        self.target_embedding = data_dictionary['decoder_embedding']
        self.decoder_sequence_length = data_dictionary['decoder_sequence_length']
        decoder_emb_inp = tf.nn.embedding_lookup(self.target_embedding, self.decoder_train_inp)
        decoder_emb_inp_time_major = tf.transpose(decoder_emb_inp, perm=[1, 0, 2])
        # decoder_sequence_length = tf.placeholder(tf.int32, shape=[None, ])

        self.dec_train_labels = data_dictionary['dec_train_labels']
        target_train_one_hot = tf.one_hot(self.dec_train_labels, constants_dictionary['TARGET_VOCAB_SIZE'], on_value=1.0, off_value=0.0)

        # processed_input_encoder = tf.transpose(enc_emb, perm=[1, 0, 2])
        initial_hidden_encoder = tf.zeros([constants_dictionary['BATCH_SIZE'], constants_dictionary['HIDDEN_LAYER_SIZE_ENCODER']])
        projection_layer = tf.layers.Dense(constants_dictionary['TARGET_VOCAB_SIZE'], use_bias=False)

        with tf.variable_scope('encoder'):
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(constants_dictionary['NUM_UNITS'])
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=self.encoder_sequence_length, dtype=tf.float32)

        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(constants_dictionary['NUM_UNITS'])
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.decoder_sequence_length)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,  output_layer=projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output

        self.prediction_output = tf.argmax(tf.nn.softmax(logits), axis=2)
        self.loss_batch = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.concat(axis=0, values=target_train_one_hot))
        self.loss = tf.reduce_mean(self.loss_batch)
