import logging

import numpy as np
import torch

from config import config
from model.corr.Configuration import Configuration
from model.corr.knob_estimator import KnobEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('storage')


class Interface:
    def __init__(self, args):
        self.args = args
        self.query_num = args.query_num
        self.queries_num = args.queries_num
        self.conf_dirpath = args.local_confdir + config['benchmark_info']['specific'] + '/'
        self.data_dirpath = args.local_datadir + config['benchmark_info']['specific'] + '/'
        self.knob_estm_model = KnobEstimator(args, self.args.query_encoder_model)
        self.c_pool = self.knob_estm_model.c_pool
        self.valid_queries_num = self.c_pool.valid_queries_num
        self.valid_queries = self.c_pool.valid_queries
        self.config_transformer = Configuration()
        self.knob_estm_model.query_encoder.load_model(config['benchmark_info']['workload'])
        self.c_pool.query_encoding = self.knob_estm_model.query_encoder.get_query_encoding(
            config['benchmark_info']['workload'])
        self.query_encoding_hit_map = {i: {} for i in range(self.valid_queries_num)}
        self.query_encoding_performance = {i: {} for i in range(self.valid_queries_num)}
        self.interval = 30

        self.knob_estm_model.load_model(
            "train_knob_estimator_" + str(self.args.num_clusters) + "_" + str(self.args.query_encoder_model)
            + "_" + config['benchmark_info']['workload'])

    def judge_uncertainty(self, sample):
        self.temp_conf = self.config_transformer.transfer_conf(sample)

        if self.args.query_encoder_model == 'KnobCF(few-shot)':
            if self.c_pool.history_confs_num < self.interval:
                struct_known = dict(
                    runtimes=np.zeros(self.queries_num, dtype=float),
                    query_need_collected=[str(i) for i in range(1, self.query_num + 1)],
                )
                return struct_known

            elif self.c_pool.history_confs_num == self.interval:
                self.update_knob_estimator()

        confs = np.repeat(self.temp_conf, self.c_pool.valid_queries_num, axis=0)
        feature = torch.Tensor(np.concatenate((self.c_pool.query_encoding, confs), axis=1)).to(self.args.device)
        out = self.knob_estm_model.predict(feature)

        struct_known = dict(
            runtimes=np.zeros(self.queries_num, dtype=float),
            query_need_collected=[str(i) for i in range(1, self.query_num + 1)],
        )

        pred_label = np.where(out.cpu().detach().numpy() > 0.1, 1, 0)
        self.temp_decimals = []
        sum = 0
        for i, binary_array in enumerate(pred_label):
            decimal = int(''.join(map(str, binary_array)), 2)
            self.temp_decimals.append(decimal)
            if decimal < 2 ** self.args.num_clusters - 1:
                if decimal not in self.query_encoding_performance[i]:
                    self.query_encoding_performance[i][decimal] = 0.0
                    self.query_encoding_hit_map[i][decimal] = 0
                if self.query_encoding_hit_map[i][decimal] >= 10:
                    value = self.query_encoding_performance[i][decimal]
                    struct_known["runtimes"][self.valid_queries[i]] = value
                    struct_known["query_need_collected"].remove(str(self.c_pool.valid_queries[i] + 1))
                    sum += 1

        logger.info("struct known len: " + str(sum))

        return struct_known

    def process(self, runtimes):
        if self.c_pool.history_confs_num < self.interval:
            self.c_pool.add_query2pool(self.temp_conf, runtimes)
        else:
            self.c_pool.add_query2pool(self.temp_conf, runtimes)
            temp_rt = runtimes[self.valid_queries]
            for i, decimal in enumerate(self.temp_decimals):
                if decimal < 2 ** self.args.num_clusters - 1:
                    self.query_encoding_performance[i][decimal] = (temp_rt[i] + self.query_encoding_performance[i][
                        decimal] * self.query_encoding_hit_map[i][decimal]) / (
                                                                          self.query_encoding_hit_map[i][decimal] + 1)
                    self.query_encoding_hit_map[i][decimal] += 1
        return

    def update_query_encoder(self):
        self.knob_estm_model.train_query_encoder(300)

    def update_knob_estimator(self):
        self.c_pool.split_train_test_dataset()
        self.knob_estm_model.train_dataset = self.c_pool.train_dataset
        self.knob_estm_model.test_dataset = self.c_pool.test_dataset
        self.knob_estm_model.train_knob_estimator(200)
        print(self.valid_queries)
        print(self.c_pool.true_time.keys())
        print(self.c_pool.queries_num)
        print(self.c_pool.valid_queries_num)
        print(self.c_pool.pass_key)

        for idx, lab in enumerate(self.c_pool.label):
            lab = np.where(lab > 0.1, 1, 0)
            decimal = int(''.join(map(str, lab)), 2)

            query_id = idx % self.valid_queries_num
            if decimal < 2 ** self.args.num_clusters - 1:
                if decimal not in self.query_encoding_performance[query_id]:
                    self.query_encoding_performance[query_id][decimal] = 0.0
                    self.query_encoding_hit_map[query_id][decimal] = 0
                self.query_encoding_performance[query_id][decimal] = (self.c_pool.true_time[self.valid_queries[query_id]][
                                                                          int(idx / self.valid_queries_num)] +
                                                                      self.query_encoding_performance[query_id][decimal]
                                                                      * self.query_encoding_hit_map[query_id][
                                                                          decimal]) / \
                                                                     (self.query_encoding_hit_map[query_id][
                                                                          decimal] + 1)
                self.query_encoding_hit_map[query_id][decimal] += 1

        self.knob_estm_model.load_model(
            "run_knob_estimator_" + str(self.args.num_clusters) + "_" + str(self.args.query_encoder_model)
            + "_" + config['benchmark_info']['workload'])
        self.knob_estimator = self.knob_estm_model.knob_estimator
