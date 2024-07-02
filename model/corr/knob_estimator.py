import logging
import math
import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, classification_report
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset
from config import config
from featurization.zero_shot_featurization.utils.fc_out_model import FcOutModel
from model.corr.ConfigurationPool import ConfigurationPool
from model.query_encoder_models.MSCN import MSCNModel
from model.query_encoder_models.QPPNet import QPPNet
from model.query_encoder_models.QueryFormerModule.QueryFormer import QueryFormerModel
from model.query_encoder_models.ZeroShotModule.ZeroShot import ZeroShot
from model.query_encoder_models.ZeroShotModule.ZeroShotSingle import ZeroShotSingle

benchmark_sp = ['tpcc', 'ycsb', 'ycsb_b', 'ycsb_c']

logger = logging.getLogger('knob_estm')
logger.setLevel(logging.DEBUG)


class KnobEstimatorModel(FcOutModel):
    def __init__(self, output_dim=None, input_dim=None, hidden_dim=512, final_out_layer=None, **final_mlp_kwargs):
        super().__init__(output_dim=output_dim, input_dim=input_dim,
                         hidden_dim=hidden_dim, final_out_layer=True, **final_mlp_kwargs)

    def forward(self, input):
        out = self.fcout(input)
        return out


class KnobEstimator:
    def __init__(self, args, query_encoder_type):
        self.args = args
        self.query_encoder_type = query_encoder_type
        if query_encoder_type == 'KnobCF(few-shot)':
            self.knob_estimator_type = str(self.args.num_clusters) + "_zero_shot_" + config['benchmark_info'][
                'workload']
        elif query_encoder_type == 'qppnet':
            self.knob_estimator_type = 'qppnet_' + config['benchmark_info']['workload']
        elif query_encoder_type == 'mscn':
            self.knob_estimator_type = 'mscn_' + config['benchmark_info']['workload']
        elif query_encoder_type == 'query_former':
            self.knob_estimator_type = str(self.args.num_clusters) + '_query_former_' + config['benchmark_info'][
                'workload']
        elif query_encoder_type == 'KnobCF':
            self.knob_estimator_type = str(self.args.num_clusters) + '_KnobCF_' + config['benchmark_info'][
                'workload']
        else:
            raise NotImplementedError

        logger.addHandler(logging.FileHandler(args.model_save_path +
                                              '/knob_estm_log/' + args.mode.split('_')[
                                                  0] + '_' + self.knob_estimator_type + '_' + self.args.type
                                              + '.log', mode='w'))

        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.args = args
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
        # self.lr = 1e-4
        self.lr = 1e-3
        self.step_size = 500
        self.gamma = 0.99
        self.labels = None

        self.c_pool = ConfigurationPool(self.args, self.args.workload)
        self.init_query_encoder()

    def train_query_encoder(self, epoch_num):
        self.query_encoder.train_query_encoder(epoch_num)

    def build_dataset(self, benchmarks, data_num=1000):
        featureTrain = None
        featureTest = None
        labelTrain = None
        labelTest = None
        for benchmark in benchmarks:
            c_pool = ConfigurationPool(self.args, benchmark)
            c_pool.query_encoding = self.query_encoder.get_query_encoding(benchmark)
            c_pool.collect_data(data_num)
            c_pool.get_train_test_dataset()
            if featureTrain is None:
                featureTrain = c_pool.featureTrain
                featureTest = c_pool.featureTest
                labelTrain = c_pool.labelTrain
                labelTest = c_pool.labelTest
            else:
                featureTrain = np.concatenate((featureTrain, c_pool.featureTrain), axis=0)
                featureTest = np.concatenate((featureTest, c_pool.featureTest), axis=0)
                labelTrain = np.concatenate((labelTrain, c_pool.labelTrain), axis=0)
                labelTest = np.concatenate((labelTest, c_pool.labelTest), axis=0)

        self.train_dataset = TensorDataset(torch.tensor(featureTrain).cuda().float(),
                                           torch.tensor(labelTrain).cuda().float())
        self.test_dataset = TensorDataset(torch.tensor(featureTest).cuda().float(),
                                          torch.tensor(labelTest).cuda().float())

    def train_knob_estimator(self, epoch_num):
        start_time = time.time()
        self.best_epoch = 0
        self.best_loss = math.inf
        for epoch in range(0, epoch_num):
            train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                       batch_size=512,
                                                       shuffle=True)
            sum_loss = 0
            for step, (feature, label) in enumerate(train_loader):
                temp_out = self.knob_estimator(feature)
                self.val_loss = self.loss_func(temp_out, label)
                sum_loss += self.val_loss.item()
                self.backward()

            if sum_loss < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = sum_loss
                self.save_model(self.args.mode.split('_')[0] + "_knob_estimator_" + self.knob_estimator_type)

            logger.info(f"epoch: {epoch}  loss: {sum_loss}")
            if epoch % 50 == 0:
                self.eval_knob_estimator(self.train_dataset)
                self.eval_knob_estimator(self.test_dataset)
            if epoch - self.best_epoch > 100:
                break

        end_time = time.time()
        logger.info(f'Time used was: {round(end_time - start_time)}s, '
                    f'{round(end_time - start_time) / 60}min, '
                    f'{round(end_time - start_time) / 3600}h')
        self.load_model(self.args.mode.split('_')[0] + "_knob_estimator_" + self.knob_estimator_type)
        self.eval_knob_estimator(self.test_dataset, True)

    def backward(self):
        self.optimizer.zero_grad()
        self.last_val_loss = self.val_loss.item()
        self.val_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def eval_knob_estimator(self, dataset, detail=False):
        print('evaluating model: ')
        self.optimizer.zero_grad()
        train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=512)
        start_time = time.time()
        for step, (feature, label) in enumerate(train_loader):
            print(feature.shape)
            pred_label = self.predict(feature)
            pred_label = np.where(pred_label.cpu().detach().numpy() > 0.1, 1, 0)
            label = label.cpu().detach().numpy()
            self.verify(label, pred_label, detail)
            break
        end_time = time.time()
        logger.info(f'Time used was: {round((end_time - start_time)*1000)}ms, '
                    f'{round(end_time - start_time) / 60}min, '
                    f'{round(end_time - start_time) / 3600}h')

    def predict(self, feature):
        out = self.knob_estimator(feature)
        return out

    def save_model(self, model_name):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        with open(os.path.join(self.save_dir, 'knob_estm', model_name + self.args.type + '.pickle'), 'wb') as file_model:
            pickle.dump(self.knob_estimator, file_model)

    def load_model(self, model_name):
        with open(os.path.join(self.save_dir, 'knob_estm', model_name + self.args.type + '.pickle'), 'rb') as f:
            self.knob_estimator = pickle.load(f)

        self.lr = 1e-4
        self.step_size = 200
        self.gamma = 0.999

        # best params
        # 5e-4 200 0.8
        self.optimizer = torch.optim.Adam(self.knob_estimator.parameters(), lr=self.lr)  # 1e-3
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size,
                                             gamma=self.gamma)  # 0.9

        logger.info(f"lr={self.lr}  step_size={self.step_size}  gamma={self.gamma}")

    def init_query_encoder(self):
        if self.query_encoder_type == 'KnobCF(few-shot)':
            self.query_encoder = ZeroShot(self.args, self.c_pool)
        elif self.query_encoder_type == 'query_former':
            self.query_encoder = QueryFormerModel(self.args, self.c_pool)
        elif self.query_encoder_type == 'KnobCF':
            self.query_encoder = ZeroShotSingle(self.args, self.c_pool)
        elif self.query_encoder_type == 'qppnet':
            self.query_encoder = QPPNet(self.args, self.c_pool)
        elif self.query_encoder_type == 'mscn':
            self.query_encoder = MSCNModel(self.args, self.c_pool)
        else:
            raise NotImplementedError

    def init_knob_estimator(self, input_dim, output_dim):

        fc_out_kwargs = dict(p_dropout=0.1, activation_class_name='LeakyReLU', activation_class_kwargs={},
                             norm_class_name='Identity', norm_class_kwargs={}, residual=True, dropout=True,
                             activation=True, inplace=True)
        final_mlp_kwargs = dict(width_factor=2, n_layers=2,
                                loss_class_name='MSELoss',  # MSELoss
                                loss_class_kwargs=dict())
        final_mlp_kwargs.update(**fc_out_kwargs)

        self.knob_estimator = KnobEstimatorModel(output_dim=output_dim, input_dim=input_dim,
                                                 hidden_dim=512, final_out_layer=True,
                                                 **final_mlp_kwargs).to(self.device)

        # best params
        # tpch: 5e-4 200 0.8
        self.optimizer = torch.optim.Adam(self.knob_estimator.parameters(), lr=self.lr)  # 1e-3
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size,
                                             gamma=self.gamma)  # 0.9

        logger.info(f"lr={self.lr}  step_size={self.step_size}  gamma={self.gamma}")

    def verify(self, label, pred_label, detail=False):
        # label = label.reshape(-1, self.output_dim)
        # print(label)
        # print(pred_label)
        # if detail:
        #     for i in range(self.num_clusters):
        #         kappa = cohen_kappa_score(label[:, i], pred_label[:, i])
        #         print('kappa系数：' + str(kappa))
        #         # model_report = classification_report(label[:, i], pred_label[:, i], digits=6, zero_division=0.0,output_dict=True)  # 分类精度报告
        #         # print('分类精度报告：\n', model_report['accuracy'])
        #         model_report = classification_report(label[:, i], pred_label[:, i], digits=6,
        #                                              zero_division=0.0)  # 分类精度报告
        #         print('分类精度报告：\n' + model_report)
        model_report = classification_report(label.reshape(-1, 1), pred_label.reshape(-1, 1), digits=6,
                                             zero_division=0.0)
        kappa = cohen_kappa_score(label.reshape(-1, 1), pred_label.reshape(-1, 1))  # kappa系数
        logger.info(f"kappa系数：{kappa}")
        logger.info(f"分类精度报告：\n{model_report}")
