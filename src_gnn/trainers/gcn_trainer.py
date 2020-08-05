from src_gnn.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time
from src_gnn.utils.utils import auc_value, f1, cmatrix


class GraphTrainer(BaseTrain):
    """
    active learning training strategy
    1. train on S, valid on S, obtain a classifier C as part of the ranker, keep a heldout test set (balanced)
    2. test samples on the pool P, get ranks for each based on uncertainty and diversity
        uncertainty: classification performance
        diversity: total KL between the existing dataset (by adding one instance) and the pool (original)
    3. get a ranked batch and add them to the existing dataset
    4. compare random selection and active selection for two or more rounds
    """

    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        # can change to super().__init__ in python3
        # the loaders (dataset) are loaded and initialized at the very beginning with the init round with mask 0,
        # and will be updated if training with more rounds
        super().__init__(sess, model, train_loader, test_loader, config, logger)
        self.sess.run(tf.tables_initializer())
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        self.eval_metric = self.config.eval_metric

    def train(self):
        """
        :return:
        """
        prev_eval_metric, prev_test_metric = 0.0, 0.0
        loss_incr_counter = 0
        optimal_epoch = 0
        optimal_eval_metric, optimal_test_metric = 0.0, 0.0
        optimal_train_loss = 100

        # traverse each epoch to train
        for cur_epoch in range(self.config.num_epochs):
            train_loss, eval_metric, eval_acc, test_metric, test_acc = self.train_epoch(prev_eval_metric, prev_test_metric)
            if cur_epoch % 20 == 0:
                print("█ Training on Epoch {} with loss {:.5f} "
                  "| eval {} {:.5f} | test {} {:.5f}".format(cur_epoch,
                                                               train_loss,
                                                               self.eval_metric, eval_metric,
                                                               self.eval_metric, test_metric
                                                             ))
            self.sess.run(self.model.increment_cur_epoch_tensor)

            ## early stopping: loss doesn't increase from previous optimal
            # if auc no change, begin the counter for early termination,
            # no change on the optimal epoch
            if train_loss > optimal_train_loss:
                loss_incr_counter += 1
            else:
                optimal_train_loss = train_loss
                loss_incr_counter = 0

            # if loss_incr_counter >= 20:
            #     print("Early stop")
            #     break

            # save the optimal eval f
            if eval_metric > optimal_eval_metric:
                optimal_epoch = cur_epoch
                optimal_eval_metric = eval_metric
                optimal_test_metric = test_metric

            prev_eval_metric = eval_metric
            prev_test_metric = test_metric

            prev_train_loss = train_loss

        print("██ Optimal train_loss is {:.5f} on epoch {} | eval {} {:.5f} | test {} {:.5f}"
              .format(optimal_train_loss, optimal_epoch,
                      self.eval_metric, optimal_eval_metric, self.eval_metric, optimal_test_metric))

        return optimal_epoch, optimal_eval_metric, optimal_test_metric

    def train_epoch(self, prev_eval_metric, prev_test_metric):
        """
        train for each round of data,
        :param prev_loss:
        :return:
        """
        # num_iter_per_epoch: number of batches for training
        train_size = self.train_loader.get_datasize()
        temp = train_size / self.config.batch_size
        self.config.num_iter_per_epoch = int(temp) + 1 if train_size % self.config.batch_size != 0 else int(temp)

        losses = []
        # traverse each batch in one epoch
        for idx in range(self.config.num_iter_per_epoch):
            loss = self.train_step(idx)
            losses.append(loss)
        train_loss = np.mean(losses)

        eval_loss, eval_acc, eval_metric = self.eval_step()

        # if loss decreases, do the test
        test_loss = 100.0
        test_loss, test_acc, test_metric = self.test_step()

        # save the epoch model
        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_metric': eval_metric,
            'test_metric': test_metric,
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(eval_metric, self.sess)  # best model

        return train_loss, eval_metric, eval_acc, test_metric, test_acc

    def train_step(self, prev_idx):
        """
        train on each batch
        :param prev_idx:
        :return:
        """
        train_inputs = next(self.train_loader.next_batch(prev_idx))
        batch_x, batch_y, batch_masks = zip(*train_inputs)

        feed_dict = {self.model.features: batch_x,
                     self.model.labels: batch_y,
                     self.model.support: self.train_loader.support_venue,
                     self.model.masks_in: batch_masks,
                     self.model.is_training: True,
                     self.model.ffd_drop: self.config.ffd_drop,
                     }

        _, loss, acc, probs, eval_metric = self.sess.run([self.model.train_step,
                                             self.model.loss,
                                             self.model.accuracy,
                                             self.model.probs,
                                             self.model.f1],
                                             feed_dict=feed_dict)
        return loss

    def eval_step(self):
        eval_ys, eval_probs, eval_preds, eval_accs, eval_losses, eval_f1s = [], [], [], [], [], []
        eval_masks = []
        eval_size = self.test_loader.get_datasize()

        temp = eval_size / self.config.batch_size
        batches = int(temp) + 1 if eval_size % self.config.batch_size != 0 else int(temp)

        for idx in range(batches):
            eval_inputs = next(self.test_loader.next_batch(idx))
            batch_x, batch_y, batch_masks = zip(*eval_inputs)
            # print(batch_x[0][1])
            # input()
            # print("# evaluating on {} samples".format(len(batch_y[0])))

            feed_dict = {self.model.features: batch_x,
                         self.model.labels: batch_y,
                         self.model.support: self.train_loader.support_venue,
                         self.model.masks_in: batch_masks,
                         self.model.is_training: False,
                         self.model.ffd_drop: 0.0,
                         }
            loss, acc, preds, probs, f1 = self.sess.run([self.model.loss,
                                                             self.model.accuracy,
                                                             self.model.preds,
                                                             self.model.probs,
                                                             self.model.f1,
                                                            ],
                                                            feed_dict=feed_dict)

            probs = [prob[1] for prob in probs]
            eval_ys.extend(batch_y)
            eval_preds.extend(preds)
            eval_probs.extend(probs)
            eval_accs.append(acc)
            eval_losses.append(loss)
            eval_f1s.append(f1)
            eval_masks.append(batch_masks)

        loss = np.mean(eval_losses)
        acc = np.mean(eval_accs)
        f1 = np.mean(eval_f1s)
        eval_probs = [np.float32(0.0) if prob != prob else prob for prob in eval_probs]

        if self.eval_metric == 'f1':
            metric = f1
        elif self.eval_metric == 'auc':
            metric = self.masked_auc(eval_probs, eval_ys, eval_masks)
        return loss, acc, metric

    def test_step(self):
        test_ys, test_probs, test_preds, test_accs, test_losses, test_f1s = [], [], [], [], [], []
        test_masks = []
        test_size = self.test_loader.get_datasize()

        temp = test_size / self.config.batch_size
        # print(test_size, self.config.batch_size, temp)
        batches = int(temp) + 1 if test_size % self.config.batch_size != 0 else int(temp)
        # batches = int(test_size / self.config.batch_size) + 1

        for idx in range(batches):
            test_inputs = next(self.test_loader.next_batch(idx))
            batch_x, batch_y, batch_masks = zip(*test_inputs)
            # print(batch_x[0][1])
            # input()
            feed_dict = {self.model.features: batch_x,
                         self.model.labels: batch_y,
                         self.model.support: self.train_loader.support_venue,
                         self.model.masks_in: batch_masks,
                         self.model.is_training: False,
                         self.model.ffd_drop: 0.0,
                         }
            loss, acc, preds, probs, f1 = self.sess.run([self.model.loss,
                                                         self.model.accuracy,
                                                         self.model.preds,
                                                         self.model.probs,
                                                         self.model.f1,
                                                         ],
                                                        feed_dict=feed_dict)
            probs = [prob[1] for prob in probs]

            test_ys.extend(batch_y)
            test_preds.extend(preds)
            test_probs.extend(probs)
            test_accs.append(acc)
            test_losses.append(loss)
            test_f1s.append(f1)
            test_masks.append(batch_masks)

        loss = np.mean(test_losses)
        acc = np.mean(test_accs)
        f1 = np.mean(test_f1s)
        test_probs = [np.float32(0.0) if prob != prob else prob for prob in test_probs]
        if self.eval_metric == 'f1':
            metric = f1
        elif self.eval_metric == 'auc':
            metric = self.masked_auc(test_probs, test_ys, test_masks)
        return loss, acc, metric

    def masked_auc(self, probs, labels, masks):
        probs = np.asarray(probs)
        labels = np.reshape(labels, (-1, self.config.num_classes))
        masks = np.reshape(masks, (-1))
        probs = [probs[i] for i in range(probs.shape[0]) if masks[i] == 1]
        labels = [labels[i] for i in range(labels.shape[0]) if masks[i] == 1]
        labels = [np.argmax(np.asarray(label)) for label in labels]
        auc = auc_value(labels, probs)
        return auc
