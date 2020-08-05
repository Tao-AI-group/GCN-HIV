import sys

sys.path.append('./')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
from src_gnn_rf_ensemble.data_loader.patient_loader_for_hiv import PatientLoader
from src_gnn_rf_ensemble.models.gat import GAT
from src_gnn_rf_ensemble.models.mlp import MLP
from src_gnn_rf_ensemble.models.gcn import GCN
from src_gnn_rf_ensemble.trainers.ensemble_gat_rf_trainer import GraphTrainer as GATRFTrainer
from src_gnn_rf_ensemble.trainers.ensemble_gcn_rf_trainer import GraphTrainer as GCNRFTrainer
from src_gnn_rf_ensemble.utils.config import get_config_from_json, update_config_by_summary, update_config_by_datasize
from src_gnn_rf_ensemble.utils.dirs import create_dirs
from src_gnn_rf_ensemble.utils.logger import Logger
from src_gnn_rf_ensemble.utils.utils import get_args
from pathlib import Path
import shutil
import pickle as pkl

import tensorflow as tf

tf.compat.v1.random.set_random_seed(1234)


def main():
    # capture the config path from the run arguments
    # then process the json configuration file

    args = get_args()
    print("getting config from {}".format(args.config))
    config, _ = get_config_from_json(args.config)
    config = update_config_by_summary(config)  # add summary and model directory
    # if remove the previous results, set -d 1
    print("If delete previous checkpoints {}".format(args.delete))
    if args.delete == '1':
        # delete existing model and summaries
        print('Deleting existing models and logs from:')
        # best_model_dir is under model dir
        print(config.summary_dir, config.model_dir, config.best_model_dir)
        path = Path(config.summary_dir)
        shutil.rmtree(path)
        path = Path(config.model_dir)
        shutil.rmtree(path)
        path = Path(config.best_model_dir)
        shutil.rmtree(path)

    config.venue_thres = int(args.threshold)

    # create the experiments dirs
    # summary dir, model dir defined in json ?
    create_dirs([config.summary_dir, config.model_dir, config.best_model_dir])

    # create your data generator to load train data

    print("Training using {}".format(config.model_version))

    if config.model_version == 'gat':
        Model = GAT
        Trainer = GATRFTrainer
    elif config.model_version == 'gcn':
        Model = GCN
        Trainer = GCNRFTrainer

    feature_path = config.exp_dir + config.ind_feature_path
    train_mask_path = config.exp_dir + config.train_mask_path
    test_mask_path = config.exp_dir + config.test_mask_path
    sex_adj_path = config.sex_adj_path
    venue_adj_path = config.venue_adj_path
    graph_feature_path = config.exp_dir + config.graph_feature_path
    psk2index_path = config.exp_dir + config.psk2index_path

    # 10 random realizations of train-test split and average, no valid is needed
    train_loader = PatientLoader(config, feature_path, sex_adj_path, venue_adj_path, train_mask_path,
                                 graph_feature_path, psk2index_path, is_train=True)
    train_loader.load()

    test_loader = PatientLoader(config, feature_path, sex_adj_path, venue_adj_path, test_mask_path, graph_feature_path,
                                psk2index_path, is_train=False)
    test_loader.load()

    # add num_iter_per_epoch to config for trainer
    config = update_config_by_datasize(config, train_loader.get_datasize(),
                                       test_loader.get_datasize(),
                                       train_loader.get_feature_size())

    # tfconfig = tf.ConfigProto(device_count={'CPU': 0})
    tfconfig = tf.ConfigProto()
    # tfconfig.gpu_options.allow_growth = True
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5

    # create tensorflow session
    with tf.Session(config=tfconfig) as sess:
        # create an instance of the model you want
        model = Model(config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = Trainer(sess, model, train_loader, test_loader, config, logger)
        # load model if exists
        # model.load(sess)
        # here you train your model
        trainer.train()

    # tester


if __name__ == '__main__':
    main()
