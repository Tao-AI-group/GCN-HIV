from src_gnn_rf_ensemble.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time
from src_gnn_rf_ensemble.utils.utils import auc_value, f1, cmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# import src_gnn_rf_ensemble.single_prediction.base.checkmate as checkmate

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
            # print("█ Training on Epoch {}".format(cur_epoch))

            train_loss, eval_metric, eval_acc, test_metric, test_acc = self.train_epoch(prev_eval_metric, prev_test_metric)
            if cur_epoch % 20 == 0:
                print("█ Training on Epoch {} with loss {:.5f} "
                  "| eval {} {:.5f} | test {} {:.5f} | ensemble auc {:.5f}".format(cur_epoch,
                                                               train_loss,
                                                               self.eval_metric, test_metric,
                                                               self.eval_metric, test_metric,
                                                               eval_metric,)
                                                             )
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

            # save the optimal eval metric
            if eval_metric > optimal_eval_metric:
                optimal_epoch = cur_epoch
                optimal_eval_metric = eval_metric
                optimal_test_metric = eval_metric
            # '''

            prev_eval_metric = eval_metric
            prev_test_metric = test_metric

            prev_train_loss = train_loss
            # time.sleep(0.4)

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

        train_losses = []
        train_probs = []
        # traverse each batch in one epoch
        for idx in range(self.config.num_iter_per_epoch):
            loss, train_probs_batch = self.train_step(idx)
            train_losses.append(loss)
            train_probs.extend(train_probs_batch)

        # ------- wrapper for random forest ---------#
        stat_features = self.train_loader.features[0]
        # from one hot 
        train_stat_labels = [np.argmax(label) for label in self.train_loader.labels[0]]
        test_stat_labels = [np.argmax(label) for label in self.test_loader.labels[0]]
        stat_masks = self.train_loader.masks[0]

        trainX, testX, trainY, testY = self.make_stat_dataset(stat_features, train_stat_labels, test_stat_labels, stat_masks)

        stat_classifier = RandomForestClassifier(max_depth=1, random_state=2)
        stat_classifier.fit(trainX, trainY)
        train_stat_preds = stat_classifier.predict(trainX)
        train_stat_probs = stat_classifier.predict_proba(trainX)

        train_probs = [train_probs[i] for i in range(len(stat_masks)) if int(stat_masks[i]) == 1]
        train_preds = [np.argmax(prob) for prob in train_probs]
        
        # combine probs and preds
        ensemble_train_features = [[train_probs[i][0], train_probs[i][1],
                                                   train_stat_probs[i][0], train_stat_probs[i][1],
                                                   train_preds[i], train_stat_preds[i]]
                                    for i in range(len(train_probs))]

        #for i, features in enumerate(ensemble_train_features):
        #    print(features, trainY[i])
        #    input()

        # ensemble model
        ensemble_classifier = RandomForestClassifier(max_depth=1, random_state=2)
        ensemble_classifier.fit(ensemble_train_features, trainY)

        # -------------------------------------------#

        train_loss = np.mean(train_losses)

        eval_loss, eval_acc, eval_metric, eval_probs = self.eval_step()

        # if loss decreases, do the test
        test_loss = 100.0
        test_loss, test_acc, test_metric, test_probs = self.test_step()

        # ------- wrapper for random forest ---------#
        test_probs = [test_probs[i] for i in range(len(stat_masks)) if int(stat_masks[i]) == 0]
        test_preds = [np.argmax(prob) for prob in test_probs]

        test_stat_probs = stat_classifier.predict_proba(testX)
        test_stat_preds = stat_classifier.predict(testX)

        ensemble_test_features = [[test_probs[i][0], test_probs[i][1],
                                                  test_stat_probs[i][0], test_stat_probs[i][1],
                                                  test_preds[i], test_stat_preds[i]]
                                    for i in range(len(test_probs))]

        ensemble_test_preds = ensemble_classifier.predict(ensemble_test_features)
        ensemble_test_probs = ensemble_classifier.predict_proba(ensemble_test_features)
        ensemble_test_probs = ensemble_test_probs[:, 1]

        ensemble_auc_score = roc_auc_score(testY, ensemble_test_probs)

        # -------------------------------------------#

        # save the epoch model
        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_metric': eval_metric,
            'test_metric': test_metric,
            'ensemble_auc': ensemble_auc_score,
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(eval_metric, self.sess)  # best model

        return train_loss, ensemble_auc_score, eval_acc, test_metric, test_acc

    def make_stat_dataset(self, features, train_labels, test_labels, masks):
        train_X, test_X, train_Y, test_Y = [], [], [], []
        for i, _ in enumerate(train_labels):
            mask = masks[i]
            if int(mask) == 0: # test
                test_Y.append(test_labels[i])
                test_X.append(features[i])
            else:
                train_Y.append(train_labels[i])
                train_X.append(features[i])
        return train_X, test_X, train_Y, test_Y


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
                     self.model.support: self.train_loader.support_sex,
                     # self.model.support: self.train_loader.support_venue,
                     self.model.masks_in: batch_masks,
                     self.model.is_training: True,
                     self.model.ffd_drop: self.config.ffd_drop,
                     }

        _, loss, acc, probs, eval_metric = self.sess.run([self.model.train_step,
                                             self.model.loss,
                                             self.model.accuracy,
                                             self.model.probs,
                                             self.model.f1,
                                             ],
                                             feed_dict=feed_dict)

        return loss, probs

    def eval_step(self):
        eval_ys, eval_probs, eval_preds, eval_accs, eval_losses, eval_f1s = [], [], [], [], [], []
        eval_masks = []
        eval_size = self.test_loader.get_datasize()

        temp = eval_size / self.config.batch_size
        batches = int(temp) + 1 if eval_size % self.config.batch_size != 0 else int(temp)

        for idx in range(batches):
            eval_inputs = next(self.test_loader.next_batch(idx))
            batch_x, batch_y, batch_masks = zip(*eval_inputs)

            feed_dict = {self.model.features: batch_x,
                         self.model.labels: batch_y,
                         self.model.support: self.train_loader.support_sex,
                         # self.model.support: self.train_loader.support_venue,
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

        if self.eval_metric == 'f1':
            metric = f1
        elif self.eval_metric == 'auc':
            metric = self.masked_auc(eval_probs, eval_ys, eval_masks)
        return loss, acc, metric, eval_probs

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
            feed_dict = {self.model.features: batch_x,
                         self.model.labels: batch_y,
                         self.model.support: self.train_loader.support_sex,
                         # self.model.support: self.train_loader.support_venue,
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
        # test_probs = [np.float32(0.0) if prob != prob else prob for prob in test_probs]
        if self.eval_metric == 'f1':
            metric = f1
        elif self.eval_metric == 'auc':
            metric = self.masked_auc(test_probs, test_ys, test_masks)
        return loss, acc, metric, test_probs


    def masked_auc(self, probs, labels, masks):
        probs = [prob[1] for prob in probs]
        probs = [np.float32(0.0) if prob != prob else prob for prob in probs]
        probs = np.asarray(probs)
        labels = np.reshape(labels, (-1, self.config.num_classes))
        masks = np.reshape(masks, (-1))
        probs = [probs[i] for i in range(probs.shape[0]) if masks[i] == 1]
        labels = [labels[i] for i in range(labels.shape[0]) if masks[i] == 1]
        labels = [np.argmax(np.asarray(label)) for label in labels]
        auc = auc_value(labels, probs)
        return auc
