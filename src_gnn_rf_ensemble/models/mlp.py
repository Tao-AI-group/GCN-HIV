"""
__author__: Xiang,Yang
"""
from src_gnn_rf_ensemble.base.base_model import BaseModel
import tensorflow as tf
from src_gnn_rf_ensemble.base.checkmate import BestCheckpointSaver
from src_gnn_rf_ensemble.models.metrics import *
tf.random.set_random_seed(1234)


class MLP(BaseModel):
    def __init__(self, config):
        # can change to super().__init__() in python3, to enable multiple inheritance
        super(MLP, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.name_scope('input'):
            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=(
                                           self.config.batch_size, self.config.num_nodes, self.config.feature_size),
                                           name='features_ph')  # input feature
            self.support = tf.sparse_placeholder(tf.float32)
            self.bias_in = tf.placeholder(dtype=tf.float32,
                                          shape=(self.config.batch_size, self.config.num_nodes, self.config.num_nodes),
                                          name='adj_bias_ph')  # bias vector from adj
            self.labels = tf.placeholder(dtype=tf.int32,
                                         shape=(self.config.batch_size, self.config.num_nodes, self.config.num_classes),
                                         name='labels_ph')  # labels
            self.masks_in = tf.placeholder(dtype=tf.int32, shape=(self.config.batch_size, self.config.num_nodes),
                                           name='masks_ph')  # consider the info or not
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())  # attention dropout
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())  # feed forward dropout
            self.is_training = tf.placeholder(dtype=tf.bool, shape=())

        # flatten to instances
        feature_reshape = tf.reshape(self.features, [-1, self.config.feature_size])
        hidden = tf.layers.dense(self.features, self.config.hid_units[0], name='dense_hidden')
        # keep [1, num_nodes, num_features] as shape, no big difference though
        # hidden = tf.layers.dense(self.features, self.config.hid_units[0], name='dense_hidden')

        self.dropout = tf.where(self.is_training, self.ffd_drop, 0.0)
        classify_feats = tf.nn.dropout(hidden, 1 - self.dropout)

        with tf.variable_scope('label_predictor'):
            # Compute logits from the output (-1) of the LSTM
            logits = tf.layers.dense(classify_feats, self.config.num_classes, name='dense_logits')

        logits_reshape = tf.reshape(logits, [-1, self.config.num_classes])
        labels_reshape = tf.reshape(self.labels, [-1, self.config.num_classes])
        masks_reshape = tf.reshape(self.masks_in, [-1])

        with tf.name_scope("loss"):
            self.loss = masked_softmax_cross_entropy(logits_reshape, labels_reshape, masks_reshape)
            self.accuracy = masked_accuracy(logits_reshape, labels_reshape, masks_reshape)
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
            #                                     (labels=labels_reshape, logits=logits_reshape))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate). \
                                                     minimize(self.loss,
                                                     global_step=self.global_step_tensor)
            self.preds = tf.argmax(logits_reshape, 1)
            self.probs = tf.sigmoid(logits_reshape)

            # correct_prediction = tf.equal(self.preds, tf.argmax(labels_reshape, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.f1 = masked_micro_f1(logits_reshape, labels_reshape, masks_reshape)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # save the 5 best ckpts

        best_ckpt_saver = BestCheckpointSaver(
            save_dir=self.config.best_model_dir,
            num_to_keep=1,
            maximize=True
        )
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = best_ckpt_saver

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "rnn":
            return tf.nn.rnn_cell.RNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.LSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("Unknown model version: {}".format(cell_type))
            return None


    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)


