"""
data loader for venue + social network

"""
import sys

sys.path.append('./')

import logging
from pathlib import Path
import string
import numpy as np
import random
import pickle as pkl
import csv
from scipy import sparse
from xlrd import open_workbook
# import tensorflow as tf
from src_gnn.utils.config import get_config_from_json, update_config_by_summary, update_config_by_datasize
from src_gnn.utils.dirs import create_dirs
from src_gnn.utils.logger import Logger
from src_gnn.utils.utils import get_args, load_adj_csv, load_adj_excel, adj_to_bias, \
    sparse_to_tuple, normalize_adj

random.seed(2010)

REDUCE_GRAPH_FEATURES = False

# PrEP and viral load relevant features are extremely associated with the determination of hiv so we remove them
# Features considered:
#       node: socio-demographic, drug usage, sex behavior, depression
#       graph features: known hiv positive neighbors, number of social venues attended, number of health venues attended
selected_attributes = [
                       'race',
                       'age_w1',
                       'black',
                       'hispanicity',
                       'smallnet_w1',
                       'education_w1',
                       'sexual_identity_w1',
                       'past12m_homeless_w1',
                       'insurance_type_w1',
                       'inconsistent_condom_w1',
                       'ever_jailed_w1',
                       'age_jailed_w1',
                       'freq_3mo_tobacco_w1',
                       'freq_3mo_alcohol_w1',
                       'freq_3mo_cannabis_w1',
                       'freq_3mo_inhalants_w1',
                       'freq_3mo_hallucinogens_w1',
                       'freq_3mo_stimulants_w1',
                       'ever_3mo_depressants_w1',
                       'num_sex_partner_drugs_w1',
                       'num_nom_sex_w1',
                       'num_nom_soc_w1',
                       'num_sex_partner_w1',
                       'num_oral_partners_w1',
                       'num_anal_partners_w1',
                       'sex_transact_money_w1',
                       'sex_transact_others_w1',
                       'depression_sum_w1',
                       ]
INTER_DEFAULT = -100


def cleaning(data):
    """ data cleaning for None values """
    for i, x in enumerate(data):
        for j, elem in enumerate(x):
            if np.isnan(elem) or elem == 'None':
                data[i][j] = INTER_DEFAULT
    return data


class PatientLoader:

    def __init__(self, config, ind_feature_path, sex_adj_path, venue_adj_path, train_mask_path,
                                 graph_feature_path, psk2index_path, is_train):
        self.config = config
        self.feature_path = ind_feature_path
        self.sex_adj_path = sex_adj_path
        self.venue_adj_path = venue_adj_path
        self.mask_path = train_mask_path
        self.graph_feature_path = graph_feature_path
        self.psk2index_path = psk2index_path
        self.labels = []  # Y
        self.features = []  # X
        self.graph_features = []  # X extension
        self.indices = []
        self.masks = []
        self.biases = []  # for GAT?
        self.adj = []
        self.psk2index = {}
        self.dataset = []
        self.datasize = 0
        self.feature_size = 0
        self.is_train = is_train

    def load(self):
        self.load_graph_features()  # graph features are loaded first so that can be added to attributes
        self.load_attributes()   # individual level features
        if self.is_train:
            self.load_psk2index()
            self.load_adj()
        self.load_mask()
        self.mask_labels()
        # standard operations of preprocessing for GAT
        self.features = self.features[np.newaxis]
        self.labels = self.labels[np.newaxis]
        self.masks = self.masks[np.newaxis]
        self.dataset = list(zip(self.features, self.labels, self.masks))
        print('num of features:', self.features.shape)
        self.datasize = self.features.shape[0]
        print("num of samples:", self.datasize)

    def load_attributes(self):
        attr_indices = []
        with open(self.feature_path) as ifile:
            ln = 0
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    header = row
                    attribute2index = {k: i for i, k in enumerate(header)}
                    attr_indices = [attribute2index[a] for a in selected_attributes]
                else:
                    # for hiv label
                    index, label_w1, label_w2, attributes = int(row[0]), int(row[1]), int(row[3]), row
                    attributes = [attributes[i] for i in attr_indices]
                    # normalize again some attributes to be used
                    attributes = self.recode_attributes(attributes)
                    attributes = [float(a) if a != "" and a != "NA" else -100.0 for a in attributes]

                    # add graph features
                    graph_attributes = self.graph_features[ln - 1]  # ln=0 is the header, so start from 1
                    attributes.extend(graph_attributes)

                    # allow diagnosis of HIV to be lagging
                    label = self.make_label_from_two_wave(label_w1, label_w2)
                    self.labels.append(label)
                    self.features.append(attributes)
                ln += 1

        self.labels = np.asarray(self.labels)
        self.labels = self.np_to_onehot(self.labels, self.config.num_classes)

        self.features = np.asarray(self.features, dtype=float)
        self.feature_size = self.features.shape[1]

    def load_mask(self):
        print('load mask from %s' % self.mask_path)
        with open(self.mask_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                self.masks.append(int(row[0]))
        self.masks = np.asarray(self.masks)

    def load_graph_features(self):
        print('load graph features from %s' % self.graph_feature_path)
        # 0.sex_centrality,     1.venue_centrality,  2.sex_num_neighbor,  3.venue_num_neighbor,
        # 4.num_social_venues,  5.num_health_venues, 6.hiv_pos_ratio,     7.hiv_neg_ratio,
        # 8.syphilis_pos_ratio, 9.syphilis_neg_ratio
        with open(self.graph_feature_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                # reduce means not consider neighbor labels, sometimes reduce perform better
                if REDUCE_GRAPH_FEATURES:
                    self.graph_features.append([float(row[i]) for i in range(6)])
                else:
                    self.graph_features.append([float(row[i]) for i in range(8)])

    def load_psk2index(self):
        ln = 0
        with open(self.psk2index_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    ln += 1
                else:
                    # print(row)
                    psk, index = row[0], row[1]
                    self.psk2index[int(psk)] = int(index)
        # print(self.psk2index)

    def load_adj(self):
        '''
        if '.csv' in self.sex_adj_path:
            self.adj_sex = load_adj_csv(self.sex_adj_path, self.psk2index)
        elif '.xls' in self.sex_adj_path:
            self.adj_sex = load_adj_excel(self.sex_adj_path, self.psk2index)
        '''
        self.adj_sex, _ = pkl.load(open(self.sex_adj_path, "rb"))

        # social w1, health w1, social w2, health w2, [As]
        self.adj_venue, _, _, _, patient2venue, _, _, _ = pkl.load(open(self.venue_adj_path, "rb"))

        # self.config.venue_thres = 30

        self.adj_venue = self.stratify_venue_matrix(self.adj_venue, self.config.venue_thres)

        num_nodes = self.config.num_nodes

        ## for GCN
        if self.config.model_version == 'gcn':
            # dense to sparse
            support_sex = sparse.csr_matrix(self.adj_sex)
            # normalize
            support_sex = normalize_adj(support_sex + sparse.eye(support_sex.shape[0]))
            # to tuple
            self.support_sex = sparse_to_tuple(support_sex)
            # dense to sparse
            support_venue = sparse.csr_matrix(self.adj_venue)
            # normalize
            support_venue = normalize_adj(support_venue + sparse.eye(support_venue.shape[0]))
            # to tuple
            self.support_venue = sparse_to_tuple(support_venue)

        ## for GAT
        if self.config.model_version == 'gat':
            self.adj_sex = self.adj_sex[np.newaxis]
            self.biases_sex = adj_to_bias(self.adj_sex, [num_nodes], nhood=1)
            self.adj_venue = self.adj_venue[np.newaxis]
            self.biases_venue = adj_to_bias(self.adj_venue, [num_nodes], nhood=1)

        print('sex and venue adjacent matrix shape:', self.adj_sex.shape, self.adj_venue.shape)

    def get_datasize(self):
        return self.datasize

    def get_dataset(self):
        return self.dataset

    def get_feature_size(self):
        return self.feature_size

    def next_batch(self, prev_idx):
        """
        the next batch of data for training
        :param prev_idx:
        :return:
        """
        b = self.config.batch_size
        upper = np.min([self.datasize, b * (prev_idx + 1)], axis=0)
        yield self.dataset[b * prev_idx: upper]

    def mask_labels(self):
        for i, _ in enumerate(self.labels):
            mask = self.masks[i]
            if int(mask) == 0:
                self.labels[i] = np.asarray([0]*self.config.num_classes)

    @staticmethod
    def np_to_onehot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    @staticmethod
    def make_label_from_two_wave(label_w1, label_w2):
        return label_w1 or label_w2

    @staticmethod
    def recode_attributes(attributes):
        new_attributes = []
        for i, name in enumerate(selected_attributes):
            value = attributes[i]
            if value == 'NA' or value == '':
                new_attributes.append(value)
                continue
            if 'sexual_identity' in name:
                if float(value) == 1 or float(value) == 3:
                    new_attributes.append(1.0)
                else:
                    new_attributes.append(0.0)
            elif name == 'education':
                if float(value) <= 2:
                    new_attributes.append(1.0)
                else:
                    new_attributes.append(0.0)
            elif 'age_jailed' in name:
                if value == '12 or younger':
                    new_attributes.append(12.0)
                else:
                    new_attributes.append(value)
            else:
                new_attributes.append(value)
        return new_attributes

    @staticmethod
    def stratify_venue_matrix(matrix, thres):
        new_matrix = []
        for row in matrix:
            new_matrix.append([1 if i >= thres else 0 for i in row])
        return np.asarray(new_matrix)
