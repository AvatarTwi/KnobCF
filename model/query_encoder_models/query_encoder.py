import math

import torch
from torch import nn

from model.corr.Configuration import Configuration

benchmark_sp = ['tpcc', 'ycsb','ycsb_b']


class QueryEncoder:
    def __init__(self, args, c_pool):
        self.args = args
        self.c_pool = c_pool
        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.args = args
        self.save_dir = args.model_save_path
        self.queries_num = args.queries_num
        self.num_clusters = args.num_clusters
        self.knob_dim = args.knob_dim
        self.save_dir = args.model_save_path
        self.best_loss = math.inf
        self.result = None
        self.loss_func = nn.MSELoss()
        self.best_epoch = 0
        self.test = {}
        self.iter = 0
        self.have_collected_data = False
        self.config_transformer = Configuration()
        self.lr = 1e-4
        self.step_size = 200
        self.gamma = 0.8
        self.labels = None

        self.mid_data_dirpath = args.model_save_path + '/mid_data/'
        self.optimizer = {}
        self.scheduler = {}

    def train_query_encoder(self, epoch_num):
        pass

    def get_query_encoding(self, benchmarks):
        pass

    def save_model(self, model_name):
        pass

    def load_model(self, model_name):
        pass