import tensorflow as tf
from src_gnn.models.metrics import *
from src_gnn.models.keras_layers import *
from src_gnn.base.base_keras_model import Model


class GCN(Model):
    def __init__(self, config):
        super(GCN, self).__init__(config)

        with tf.name_scope('input'):
            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=(self.config.batch_size, self.config.num_nodes, self.config.feature_size),
                                           name='features_ph')  # input feature
            self.support = tf.sparse_placeholder(tf.float32)  # adj for gcn
            self.labels = tf.placeholder(dtype=tf.int32,
                                         shape=(self.config.batch_size, self.config.num_nodes, self.config.num_classes),
                                         name='labels_ph')  # labels
            self.masks_in = tf.placeholder(dtype=tf.int32, shape=(self.config.batch_size, self.config.num_nodes),
                                           name='masks_ph')  # consider the info or not
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')  # feed forward dropout
            self.is_training = tf.placeholder(dtype=tf.bool, shape=())

            self.initializer = tf.random_normal_initializer(stddev=0.1)

        self.inputs = tf.reshape(self.features, [-1, self.config.feature_size])
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.config.l2_coef * tf.nn.l2_loss(var)

        # Cross entropy error
        logits_reshape = tf.reshape(self.outputs, [-1, self.config.num_classes])
        labels_reshape = tf.reshape(self.labels, [-1, self.config.num_classes])
        masks_reshape = tf.reshape(self.masks_in, [-1])

        self.loss += masked_softmax_cross_entropy(logits_reshape, labels_reshape, masks_reshape)

        self.probs = tf.sigmoid(logits_reshape)
        self.preds = tf.cast(tf.argmax(logits_reshape, 1), tf.int32)
        self.accuracy = masked_accuracy(logits_reshape, labels_reshape, masks_reshape)
        self.f1 = masked_micro_f1(logits_reshape, labels_reshape, masks_reshape)


    def _accuracy(self):
        pass

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.config.feature_size,
                                            output_dim=self.config.hidden_size,
                                            support=self.support,
                                            act=tf.nn.relu,
                                            dropout=self.ffd_drop,
                                            sparse_inputs=False,
                                            is_training=self.is_training,
                                            ))

        self.layers.append(GraphConvolution(input_dim=self.config.hidden_size,
                                            output_dim=self.config.num_classes,
                                            support=self.support,
                                            act=lambda x: x,
                                            dropout=self.ffd_drop,
                                            sparse_inputs=False,
                                            is_training=self.is_training,
                                           ))

    def predict(self):
        return tf.nn.softmax(self.outputs)
