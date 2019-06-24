
import re
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from modelE2D import E2D

abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)
'''
To Do :
    - save model
    - predict
    - argparse
    - 
'''



def train(data, embedding, vocab):

    print("Train function Started :")
    source_data_idx, target_data_idx = data[0], data[1]
    encoder_embedding, decoder_embedding = embedding[0], embedding[1]
    source_vocab, target_vocab = vocab[0], vocab[1]

    # print("Data  : ", source_data_idx)
    # print("Embedding : ", encoder_embedding)
    # print("Vocab : ", source_vocab)

    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    NUM_UNITS = 256
    SOURCE_VOCAB_SIZE = len(source_vocab)
    TARGET_VOCAB_SIZE = len(target_vocab)
    HIDDEN_LAYER_SIZE_ENCODER = 256

    encoder_train_inp = tf.placeholder(tf.int32, shape=[None, None], name='inputs_encoder')
    decoder_train_inp = tf.placeholder(tf.int32, shape=[None, None], name='dec_train_inputs')
    dec_train_labels = tf.placeholder(tf.int32, shape=[None, None], name='dec_train_labels')

    encoder_sequence_length = tf.placeholder(tf.int32, shape=[None, ])
    decoder_sequence_length = tf.placeholder(tf.int32, shape=[None, ])

    # processed_input_encoder = tf.transpose(enc_emb, perm=[1, 0, 2])
    # initial_hidden_encoder = tf.zeros([BATCH_SIZE, HIDDEN_LAYER_SIZE_ENCODER])
    # projection_layer = tf.layers.Dense(TARGET_VOCAB_SIZE, use_bias=False)

    dataset = tf.data.Dataset.from_tensor_slices((encoder_train_inp, decoder_train_inp)).batch(BATCH_SIZE).repeat()
    iterator = dataset.make_initializable_iterator()
    source_sent, target_sent = iterator.get_next()

    constants_dictionary = {
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'TARGET_VOCAB_SIZE': TARGET_VOCAB_SIZE,
        'SOURCE_VOCAB_SIZE': SOURCE_VOCAB_SIZE,
        'HIDDEN_LAYER_SIZE_ENCODER': HIDDEN_LAYER_SIZE_ENCODER,
        'NUM_UNITS': NUM_UNITS
    }

    data_dictionary = {
        'encoder_train_inp': encoder_train_inp,
        'decoder_train_inp': decoder_train_inp,
        'dec_train_labels': dec_train_labels,
        'encoder_sequence_length': encoder_sequence_length,
        'decoder_sequence_length': decoder_sequence_length,
        'encoder_embedding': encoder_embedding,
        'decoder_embedding': decoder_embedding
    }

    model = E2D(constants_dictionary, data_dictionary)
    optimizer = tf.train.AdamOptimizer().minimize(model.loss)

    with tf.Session() as sess:
        print("Training session started ....")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer, feed_dict={encoder_train_inp: source_data_idx, decoder_train_inp: target_data_idx})

        for epoch in range(NUM_EPOCHS):
            t_source_sent, t_target_sent = sess.run([source_sent, target_sent])

            t_source_sent_length = [np.where(temp==2)[0][0] if len(np.where(temp==2)[0])>0 else 0 for temp in t_source_sent]
            t_target_sent_length = [np.where(temp==2)[0][0] if len(np.where(temp==2)[0])>0 else 0 for temp in t_target_sent]
            # t_target_sent_label = [temp[:np.where(temp==2)[0][0]] for temp in t_target_sent]

            max_target_sent_len = max(t_target_sent_length)
            label_list = list()
            for _temp in t_target_sent:
                label_list.append(_temp[:max_target_sent_len])

            feed_dict = {encoder_train_inp: t_source_sent, decoder_train_inp: t_target_sent, dec_train_labels: label_list,
                         encoder_sequence_length: t_source_sent_length, decoder_sequence_length: t_target_sent_length}
            sess.run(optimizer, feed_dict=feed_dict)

    print("Training session ended ....")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data_dir', type=str, default='mnist_data/', help='Directory for storing input data')
    # parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    # parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    # parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    # parser.add_argument('-e', '--epochs', type=int, default=30, help='How many epochs to run in total?')
    # parser.add_argument('-b', '--batch_size', type=int, default=50, help='Batch size during training per GPU')
    # parser.add_argument('-v', '--val_size', type=int, default=5000)
    # args = parser.parse_args()

    # pretty print args
    # print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

    # data = mnist_data.read_data_sets(args.data_dir, one_hot=True, reshape=False, validation_size=args.val_size)
    # data = input_data.read_data_sets(args.data_dir, one_hot=True, reshape=False, validation_size=args.val_size)

    # if not os.path.exists(args.ckpt_dir):
    #     os.makedirs(args.ckpt_dir)
    print("Main code Started :")
    # kan_emb_path = "C:/Users/shashidhara.kc/Documents/Documents_Nalakku/kalli/Learning TensorFlow Itay Lieder/Learning-TensorFlow-master/MyExperiments/kan_embedding.npy"
    # eng_emb_path = "C:/Users/shashidhara.kc/Documents/Documents_Nalakku/kalli/Learning TensorFlow Itay Lieder/Learning-TensorFlow-master/MyExperiments/eng_embedding.npy"
    kan_embedding = np.load("../kan_embedding.npy")
    eng_embedding = np.load("../eng_embedding.npy")
    embedding_list = list([kan_embedding, eng_embedding])
    source_data_idx_load = pickle.load(open("../sourceDataIdx.p", "rb"))
    target_data_idx_load = pickle.load(open("../targetDataIdx.p", "rb"))
    data_list = list([source_data_idx_load, target_data_idx_load])
    source_data_dictionary_load = pickle.load(open("../sourceVocabDict.p", "rb"))
    target_data_dictionary_load = pickle.load(open("../targetVocabDict.p", "rb"))
    vocab_list = list([source_data_dictionary_load, target_data_dictionary_load])

    print("Call main code Started :")

    train(data_list, embedding_list, vocab_list)
