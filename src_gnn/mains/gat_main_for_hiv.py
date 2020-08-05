import sys

sys.path.append('./')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
from src_gnn.data_loader.patient_loader_for_hiv import PatientLoader
from src_gnn.models.gat import GAT
from src_gnn.trainers.gat_trainer import GraphTrainer as GatTrainer
from src_gnn.utils.config import get_config_from_json, update_config_by_summary, update_config_by_datasize
from src_gnn.utils.dirs import create_dirs
from src_gnn.utils.logger import Logger
from src_gnn.utils.utils import get_args
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

    Model = GAT
    Trainer = GatTrainer

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

    tfconfig = tf.ConfigProto()
    # specify GPU usage if using GPU
    #tfconfig.gpu_options.allow_growth = True
    #tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4

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
