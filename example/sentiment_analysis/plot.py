import sys
import os

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../../src")
from tcn import *
import pickle

import tensorflow as tf


class text_classifier():
    def __init__(self):
        self.sess = tf.Session()
        self.batch_size = 200
        self.learning_rate = 0.001
        self.steps = 3000
        pass

    def load_data(self, train_path, datatype, eval_path=None, header=True, preprocessfunc=None, label='label'):
        self.label = label
        self.traindata = tf.contrib.data.CsvDataset(train_path, datatype, header=header, na_value='')
        self.traindata = self.traindata.repeat().shuffle(100, seed=1024).batch(self.batch_size,
                                                                               drop_remainder=True).map(preprocessfunc)
        self.iterator = tf.data.Iterator.from_structure(self.traindata.output_types,
                                                        self.traindata.output_shapes,
                                                        output_classes=self.traindata.output_classes)
        self.datatensor = self.iterator.get_next()
        self.training_init_op = self.iterator.make_initializer(self.traindata)
        if not eval_path:
            self.eval_init_op = self.iterator.make_initializer(self.traindata)
        else:
            self.evaldata = tf.contrib.data.CsvDataset(eval_path, datatype, header=header, na_value='')
            self.evaldata = self.evaldata.batch(self.batch_size, drop_remainder=True).map(preprocessfunc)
            self.eval_init_op = self.iterator.make_initializer(self.evaldata)

        pass

    def test(self):
        pass

    def model(self):
        input_tensor, seq_len = tf.contrib.feature_column.sequence_input_layer(self.datatensor,
                                                                               feature_columns=self.feature_columns[
                                                                                   'text'])
        input_tensor = tf.expand_dims(input_tensor, -1)
        out = tcn_layer(input_tensor, 2, filter_size=5)
        out = tcn_block(out, input_tensor, use_conv=True)
        out = tf.reduce_mean(out, axis=-1)
        # we should have dimension[N,S,embedding_size] til here
        cell = tf.nn.rnn_cell.LSTMCell(50, activation='tanh')
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, out,
                                           initial_state=initial_state,
                                           sequence_length=seq_len,
                                           dtype=tf.float32)
        self.tempout = state
        self.output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2')(
            state[1])

    def _compile(self):
        self.metric_dict = {}
        self.label_tensor = tf.feature_column.input_layer(self.datatensor,
                                                          feature_columns=self.feature_columns['label'])
        self.metric_dict['CrossEntropy'] = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(self.label_tensor, self.output, 5))
        self.prediction = tf.cast(self.output > 0.5, tf.float32)
        self.metric_dict['ACC'] = tf.metrics.accuracy(self.label_tensor, self.prediction)
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = self.metric_dict['CrossEntropy'] + self.reg_loss
        self.metric_dict['AUC'] = tf.metrics.auc(self.label_tensor, self.output)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        init_op = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        print('initalize the variables')
        self.sess.run(init_op)
        self.sess.run(init_l)

        with self.sess.as_default():
            print('Initize table')
            self.sess.run(tf.tables_initializer())
            # print('Write tf graph')
            # self.writer = tf.summary.FileWriter("./tmp", self.sess.graph)
            # self.writer.close()
        pass

    def plot(self):
        self.loss
        pass

    def eval_func(self):
        with self.sess.as_default():
            self.sess.run(self.eval_init_op)
            reslst = []
            resdict = {}
            while 1:
                try:
                    reslst.append(self.sess.run(self.metric_dict))
                except tf.errors.OutOfRangeError:
                    break
            for key in self.metric_dict:
                resdict[key] = sum([x[key] for x in reslst]) / len(reslst)
            print('evalutation:', resdict)
            self.sess.run(self.training_init_op)
        pass

    def train(self):
        with self.sess.as_default():
            self.sess.run(self.training_init_op)
            for i in range(1, self.steps + 1):
                step = self.sess.run([self.train_step, self.metric_dict, self.label_tensor, self.prediction])
                if not i % 100:
                    print(i, 'training:', step[1])
                    if not i % 3000:
                        self.eval_func()

    def get_feature_columns(self):
        vocablist = pickle.load(open('vocablist.pkl', 'rb'))
        print(len(vocablist))
        init = tf.contrib.framework.load_embedding_initializer(
            ckpt_path='embedding_lookup',
            embedding_tensor_name='glove.twitter.27B.50d.txt',
            new_vocab_size=len(vocablist),
            embedding_dim=50,
            old_vocab_file='vocablist.txt',
            new_vocab_file='vocablist.txt'
        )

        text = tf.contrib.feature_column.sequence_categorical_column_with_vocabulary_list('text', vocablist.keys())
        text = tf.feature_column.embedding_column(text, 50, initializer=init, trainable=False)
        label = tf.feature_column.numeric_column(self.label)
        self.feature_columns = dict(
            text=text,
            label=label,
        )
        pass


if __name__ == '__main__':
    tfmodel = text_classifier()
    preprocessfunc = lambda x, y: dict(zip(['text', 'sentiment'], [tf.string_split(x), y]))
    tfmodel.load_data('Sentiment_cleaned.csv', [tf.string, tf.float32],
                      # eval_path='../../data/regression/londonreview_eval.csv',
                      header=True,
                      preprocessfunc=preprocessfunc,
                      label='sentiment')
    tfmodel.get_feature_columns()
    tfmodel.model()
    tfmodel._compile()
    tfmodel.train()
    pass
