import tensorflow as tf
import src_gnn_rf_ensemble.base.checkmate as checkmate

class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, auc, sess):
        # print("Saving models...")
        # self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        self.saver.handle(auc, sess, self.global_step_tensor)   ####
        # print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        best_checkpoint = checkmate.get_best_checkpoint(self.config.best_model_dir, select_maximum_value=True)
        if best_checkpoint:
            tf.train.Saver().restore(sess, best_checkpoint)
            print("Best models loaded")
        else:
            print("Best models doesn't exist")
        # latest_checkpoint = tf.train.latest_checkpoint(self.config.model_dir)
        # if latest_checkpoint:
        #     print("Loading models checkpoint {} ...\n".format(latest_checkpoint))
        #     self.saver.restore(sess, latest_checkpoint)
        #     # print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainers
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

