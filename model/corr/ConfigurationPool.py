import os
import pickle

import lightgbm as lgb
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from torch.utils.data import TensorDataset

from config import config
from featurization.benchmark_tools.utils import load2json
from model.corr.Configuration import Configuration
from model.corr.Statistic import Statistic
from utils.path_utils import PathUtils

benchmark_sp = ['tpcc', 'ycsb','ycsb_b']


class ConfigurationPool:
    def __init__(self, args, workload):
        self.args = args
        self.workload = workload
        self.queries_num = args.queries_num
        self.history_confs = None
        self.history_confs_num = 0
        self.history_runtimes = dict()
        self.history_label = dict()
        self.label = None
        self.feature = None
        self.query_encoding = None
        self.cluster_points = None
        self.mu_sigma = None
        self.save_dir = args.model_save_path
        self.num_clusters = args.num_clusters
        self.knob_dim = args.knob_dim
        self.config_transformer = Configuration()
        self.pass_key, self.statis_avg, self.statis_diff = Statistic.data_statistics_queries(workload)
        self.valid_queries = [i for i in range(args.queries_num) if i not in self.pass_key]
        self.valid_queries_num = len(self.valid_queries)
        self.device = torch.device('cuda:0')

    def collect_data(self, data_num=1000):
        iter = 0
        files = PathUtils.return_files_in_directory(self.args.local_confdir)
        files = [file for file in files if self.workload in file]
        data_paths = [self.args.local_datadir+file for file in files]
        conf_paths = [self.args.local_confdir+file for file in files]
        for idx, path in enumerate(conf_paths):
            if self.workload in benchmark_sp:
                while os.path.exists(os.path.join(path, 'knob_data' + str(iter) + '.pickle')) \
                        and os.path.exists(os.path.join(data_paths[idx], 'statis' + str(iter) + '.pickle')):

                    with open(os.path.join(data_paths[idx], 'statis' + str(iter) + '.pickle'), 'rb') as f:
                        file = pickle.load(f)
                        try:
                            run_times = np.array(file)[:, 0]
                        except:
                            run_times = []
                    with open(os.path.join(path, 'knob_data' + str(iter) + '.pickle'), 'rb') as f:
                        conf = pickle.load(f)
                        conf = self.config_transformer.transfer_conf(conf)
                    self.add_query2pool(conf, run_times)
                    iter += 1
                    if len(self.history_confs) == data_num:
                        break
            else:
                while os.path.exists(os.path.join(path, 'knob_data' + str(iter) + '.pickle')) \
                        and os.path.exists(os.path.join(data_paths[idx], 'serverlog' + str(iter) + '.txt')):

                    run_times = load2json(os.path.join(data_paths[idx], 'serverlog' + str(iter) + '.txt'))
                    with open(os.path.join(path, 'knob_data' + str(iter) + '.pickle'), 'rb') as f:
                        conf = pickle.load(f)
                        conf = self.config_transformer.transfer_conf(conf)
                    self.add_query2pool(conf, run_times)
                    iter += 1
                    if len(self.history_confs) == data_num:
                        break

    def add_query2pool(self, conf, runtime):
        valid = self.add_runtime2pool(runtime)
        if valid:
            self.add_conf2pool(conf)
            self.history_confs_num += 1
            return True
        return False

    def add_runtime2pool(self, input):
        if len(self.history_runtimes) == 0:
            for i in range(len(input)):
                self.history_runtimes[i] = []
        if len(input) == len(self.history_runtimes):
            for i, item in enumerate(input):
                self.history_runtimes[i].append(item)
            return True
        else:
            return False

    def add_conf2pool(self, conf):
        if self.history_confs is None:
            self.history_confs = conf
        else:
            self.history_confs = np.concatenate((self.history_confs, conf), axis=0)

    # present query with knob importance
    def get_query_presentation(self):
        self.query_presentation = None
        for key, item in self.history_runtimes.items():
            if key in self.pass_key:
                continue
            y = item
            params = {
                'verbosity': -1,
                'random_state': 42
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(self.history_confs, y)

            # 获取特征重要性
            feature_importance = np.array(model.feature_importances_).reshape(1, -1)
            feature_importance = Normalizer().fit_transform(feature_importance)
            if self.query_presentation is None:
                self.query_presentation = feature_importance
            else:
                self.query_presentation = np.concatenate((self.query_presentation, feature_importance), axis=0)
        return

    def get_feature(self):
        for conf in self.history_confs:
            temp_conf = np.repeat(conf.reshape(1, -1), self.query_encoding.shape[0], axis=0)
            if self.feature is None:
                self.feature = np.concatenate((self.query_encoding, temp_conf), axis=1)
            else:
                feature = np.concatenate((self.query_encoding, temp_conf), axis=1)
                self.feature = np.concatenate((self.feature, feature), axis=0)

    def get_classify_result(self):
        for key, item in self.history_runtimes.items():
            if key not in self.pass_key:
                # 使用高斯混合模型将数据点聚类
                X = np.array(item).reshape(-1, 1)
                gmm = GaussianMixture(init_params='k-means++', n_components=self.num_clusters, covariance_type='full')
                gmm.fit(X)

                cluster_centers = np.sort(gmm.means_[:, -1]).reshape(1, -1)
                if self.cluster_points is None:
                    self.cluster_points = cluster_centers
                else:
                    self.cluster_points = np.concatenate((self.cluster_points, cluster_centers),
                                                         axis=0)
        return

    def get_label(self):
        self.get_classify_result()
        data = [runtimes for key, runtimes in self.history_runtimes.items() if key not in self.pass_key]
        statis_diff = [statis for key, statis in self.statis_diff.items() if statis != 0]
        data = np.array(data)
        for l in range(data.shape[1]):
            value = np.abs(np.array(data[:, l]).reshape(-1, 1) - self.cluster_points)
            for i in range(value.shape[0]):
                # 使用逻辑运算符和逐元素的比较进行条件判断
                condition = (0 < value[i, :]) & (value[i, :] < statis_diff[i])
                value[i, :] = np.where(condition, 0, 1)
            if self.label is None:
                self.label = value
            else:
                self.label = np.concatenate((self.label, value), axis=0)

    def split_train_test_dataset(self):
        self.get_feature()
        self.get_label()
        self.true_time = {key:runtimes for key, runtimes in self.history_runtimes.items() if key not in self.pass_key}

        self.featureTrain, self.featureTest, self.labelTrain, self.labelTest \
            = train_test_split(self.feature, self.label, train_size=0.8, random_state=42)

        print(self.feature.shape)
        print(self.label.shape)

        self.train_dataset = TensorDataset(torch.tensor(self.featureTrain).cuda().float(),
                                           torch.tensor(self.labelTrain).cuda().float())
        self.test_dataset = TensorDataset(torch.tensor(self.featureTest).cuda().float(),
                                          torch.tensor(self.labelTest).cuda().float())

    def get_train_test_dataset(self):
        self.get_feature()
        self.get_label()
        self.featureTrain, self.featureTest, self.labelTrain, self.labelTest \
            = train_test_split(self.feature, self.label, train_size=0.8, random_state=42)

    def _get_label(self):
        for key, item in self.history_runtimes.items():
            if key not in self.pass_key:
                # 使用高斯混合模型将数据点聚类
                X = np.array(item).reshape(-1, 1)
                gmm = GaussianMixture(init_params='k-means++', n_components=self.num_clusters, covariance_type='full')
                gmm.fit(X)
                cluster_centers = gmm.means_
                proba = gmm.predict_proba(X)
                value = np.where(proba > 0.1, 0, 1)[:, np.argsort(np.array(cluster_centers).flatten())]
                if self.label is None:
                    self.label = value
                else:
                    self.label = np.concatenate((self.label, value), axis=0)
        return
