import tensorflow as tf

"""
Train on a multi-round framework (simulate active learning)
"""
class BaseTrain:
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.sess.run(self.init)

    def train(self):
        """
        :return: 
        """
        raise NotImplementedError

    def train_patch(self):
        """
        train on a round of training set
        :return:
        """
        raise NotImplementedError

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
