import logging
import math
import os
import pickle
import time

import numpy as np
import torch
from torch.optim import lr_scheduler

from config import config
from dataset.load_json_plan import load2json_plan
from featurization.benchmark_tools.postgres.parse_plan import parse_plans
from featurization.benchmark_tools.utils import load_attr_table, load_database_stats
from featurization.zero_shot_featurization.dataset_creation import dataset_creation
from model.corr.ConfigurationPool import ConfigurationPool
from model.corr.Statistic import Statistic
from model.query_encoder_models.ZeroShotModule.postgres_zero_shot import PostgresZeroShotModel
from model.query_encoder_models.query_encoder import QueryEncoder
from utils.constant import queries_num

benchmark_sp = ['tpcc', 'ycsb', 'ycsb_b', 'ycsb_c']

logger = logging.getLogger(KnobCF')
logger.setLevel(logging.DEBUG)


class ZeroShot(QueryEncoder):
    def __init__(self, args, c_pool=None):
        super().__init__(args, c_pool=None)
        self.c_pool = c_pool

        logger.addHandler(
            logging.FileHandler(args.model_save_path +
                                '/query_encoder_log/' + args.mode.split('_')[0] + '_zero_shot_' + self.args.workload
                                + '_' + self.args.type
                                + '.log', mode='w'))
        self.args = args
        self.workload = self.args.workload
        self.benchmarks = ['tpch', 'tpcc', 'job', 'ycsb', 'ycsb_b', 'ycsb_c']
        self.valid_range = {}
        self.valid_queries_each = {}
        pass_keys_all = []
        sum = 0
        for benchmark in self.benchmarks:
            pass_keys, _, _ = Statistic.data_statistics_queries(workload=benchmark)
            temp = sum - len(pass_keys)
            pass_keys_all.extend([i + sum for i in pass_keys])
            sum += queries_num[benchmark]
            self.valid_range[benchmark] = (temp, sum - len(pass_keys_all))
            self.valid_queries_each[benchmark] = [i for i in range(queries_num[benchmark]) if i not in pass_keys]
        self.valid_queries = [i for i in range(sum) if i not in pass_keys_all]

    def get_valid_queries(self, benchmarks):
        sum = 0
        pass_keys_all = []
        for benchmark in benchmarks:
            pass_keys, _, _ = Statistic.data_statistics_queries(workload=benchmark)
            pass_keys_all.extend([i + sum for i in pass_keys])
            sum += queries_num[benchmark]
        valid_queries = [i for i in range(sum) if i not in pass_keys_all]
        return valid_queries

    def parse_benchmark(self):
        benchmarks = []
        if self.args.type == 'on':
            benchmarks = []
            if config['benchmark_info']['workload'] != 'ycsb_b':
                benchmarks.append('ycsb_b')
            if config['benchmark_info']['workload'] == 'ycsb_b':
                benchmarks.append('ycsb')
            # benchmarks.append(self.workload)
        else:
            for b in ['tpch', 'tpcc', 'job', 'ycsb']:
                if b == config['benchmark_info']['workload']:
                    continue
                benchmarks.append(b)
        parsed_runs = self.parse_run(benchmarks)
        parsed_runs['parsed_plans'] = list(np.array(parsed_runs['parsed_plans'])[self.get_valid_queries(benchmarks)])
        self.result, self.feature_statistics = dataset_creation(parsed_runs)
        labels = []
        for benchmark in benchmarks:
            c_pool = ConfigurationPool(self.args, benchmark)
            c_pool.collect_data()
            c_pool.get_query_presentation()
            labels.extend(c_pool.query_presentation)
        self.labels = torch.tensor(np.array(labels)).float().to('cuda:0')

    def train_query_encoder(self, epoch_num):
        self.parse_benchmark()
        self.init_query_encoder()
        self.best_epoch = 0
        self.best_loss = math.inf

        for epoch in range(epoch_num):
            out = self.query_encoder()
            self.val_loss = self.loss_func(out, self.labels)
            self.backward()

            if self.val_loss.item() < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = self.val_loss.item()
                self.save_model(self.workload)
            if epoch - self.best_epoch > 100:
                break
            logger.info(f"epoch: {epoch}  loss: {self.val_loss.item()}")
        logger.info(f"best epoch: {self.best_epoch}  best loss: {self.best_loss}")
        self.eval_query_encoder(self.workload)

    def eval_query_encoder(self, benchmark, epoch_num=1):
        start_time = time.time()
        parsed_runs = self.parse_run([benchmark])
        parsed_runs['parsed_plans'] = list(np.array(parsed_runs['parsed_plans'])[self.valid_queries_each[benchmark]])
        self.result, self.feature_statistics = dataset_creation(parsed_runs)
        self.query_encoder.update_result(self.result)
        self.c_pool.collect_data()
        self.c_pool.get_query_presentation()
        self.label = torch.tensor(self.c_pool.query_presentation).float().to(self.device)
        for epoch in range(epoch_num):
            self.optimizer.zero_grad()
            out = self.query_encoder()
            self.val_loss = self.loss_func(out, self.label)
            self.backward()
            if self.val_loss.item() < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = self.val_loss.item()
                self.save_model(self.workload)
            if epoch - self.best_epoch > 100:
                break
            logger.info(f"epoch: {epoch}  loss: {self.val_loss.item()}")

        end_time = time.time()
        logger.info(f'Time used was: {round(end_time - start_time)}s, '
                    f'{round(end_time - start_time) / 60}min, '
                    f'{round(end_time - start_time) / 3600}h')
        logger.info(f"best epoch: {self.best_epoch}  best loss: {self.best_loss}")

    def get_query_encoding(self, benchmark):
        parsed_runs = self.parse_run([benchmark])
        parsed_runs['parsed_plans'] = list(np.array(parsed_runs['parsed_plans'])[self.valid_queries_each[benchmark]])
        self.result, self.feature_statistics = dataset_creation(parsed_runs)
        self.query_encoder.update_result(self.result)
        self.query_encoder.test = True
        self.optimizer.zero_grad()
        query_encoding = self.query_encoder().cpu().detach().numpy()
        return query_encoding

    def backward(self):
        self.optimizer.zero_grad()
        self.last_val_loss = self.val_loss.item()
        self.val_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def init_query_encoder(self):
        fc_out_kwargs = dict(p_dropout=0.1, activation_class_name='LeakyReLU', activation_class_kwargs={},
                             norm_class_name='Identity', norm_class_kwargs={}, residual=True, dropout=True,
                             activation=True, inplace=True)
        final_mlp_kwargs = dict(width_factor=2, n_layers=3,
                                loss_class_name='MSELoss',  # MSELoss
                                loss_class_kwargs=dict())

        tree_node_pass_dim = 256
        tree_layer_kwargs = dict(width_factor=1, n_layers=2, hidden_dim=tree_node_pass_dim)
        node_type_kwargs = dict(width_factor=1, n_layers=2, one_hot_embeddings=True, max_emb_dim=32,
                                drop_whole_embeddings=False, output_dim=tree_node_pass_dim)

        pg_zero_shot_dim = tree_node_pass_dim

        final_mlp_kwargs.update(**fc_out_kwargs)
        tree_layer_kwargs.update(**fc_out_kwargs)
        node_type_kwargs.update(**fc_out_kwargs)
        output_dim = self.args.knob_dim

        self.query_encoder = PostgresZeroShotModel(device="cuda:0",
                                                   input_dim=pg_zero_shot_dim,
                                                   hidden_dim=256,
                                                   output_dim=output_dim,
                                                   final_mlp_kwargs=final_mlp_kwargs,
                                                   node_type_kwargs=node_type_kwargs,
                                                   tree_layer_kwargs=tree_layer_kwargs,
                                                   tree_layer_name='GATConv',
                                                   plan_featurization_name='PostgresTrueCardDetail',
                                                   feature_statistics=self.feature_statistics,
                                                   result=self.result).to(self.device)

        lr = 1e-3
        step_size = 500
        gamma = 0.9
        self.optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=lr)  # 1e-3
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size,
                                             gamma=gamma)  # 0.9

        logger.info(f"lr={lr}  step_size={step_size}  gamma={gamma}")

    def save_model(self, model_name=''):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        with open(os.path.join(self.save_dir, 'query_encoder', 'query_encoder_zeroshot_' + model_name + self.args.type + '.pickle'),
                  'wb') as file_model:
            pickle.dump(self.query_encoder, file_model)

    def load_model(self, model_name=''):
        with open(os.path.join(self.save_dir, 'query_encoder', 'query_encoder_zeroshot_' + model_name + self.args.type + '.pickle'),
                  'rb') as f:
            self.query_encoder = pickle.load(f)
            self.query_encoder.update_result(self.result)

        # best params
        # 5e-4 200 0.8
        self.optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=self.lr)  # 1e-3
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size,
                                             gamma=self.gamma)  # 0.9

        logger.info(f"lr={self.lr}  step_size={self.step_size}  gamma={self.gamma}")

    def parse_run(self, benchmarks):

        explain_datas = []
        attr_tables = {}
        database_statses = {}

        for benchmark in benchmarks:
            path = os.path.join(self.args.local_datadir + benchmark + '4base' + '/', 'serverlog0.txt')
            explain_data = load2json_plan(path)[:queries_num[benchmark]]

            attr_table = load_attr_table(benchmark)
            database_stats = load_database_stats(benchmark)

            explain_datas.extend(explain_data)
            attr_tables.update(attr_table)
            if database_statses == {}:
                database_statses = database_stats
            else:
                for key in database_statses.keys():
                    database_statses[key].extend(database_stats[key])
        run_stats = dict(query_list=explain_datas,
                         database_stats=database_statses,
                         total_time_secs=[data['Actual Total Time'] for data in explain_datas],
                         attr_table=attr_tables)

        parsed_runs, stats = parse_plans(run_stats)
        return parsed_runs
