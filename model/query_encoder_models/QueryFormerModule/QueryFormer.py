import logging
import math
import os
import time

from torch.optim import lr_scheduler
from model.query_encoder_models.QueryFormerModule.model import QueryFormer
from model.query_encoder_models.QueryFormerModule.plan_tree_dataset import PlanTreeDataset
from model.query_encoder_models.query_encoder import QueryEncoder
from model.database_util import *

logger1 = logging.getLogger('query_former')
logger1.setLevel(logging.DEBUG)


class QueryFormerModel(QueryEncoder):
    def __init__(self, args, c_pool):
        super().__init__(args, c_pool)
        logger1.addHandler(
            logging.FileHandler(args.model_save_path +
                                '/query_encoder_log/' + args.mode.split('_')[0] + '_query_former_' + self.args.workload
                                + '.log', mode='w'))
        self.lr = 1e-3
        self.step_size = 500
        self.gamma = 0.99
        self.args = args
        self.workload = args.workload
        self.queries_num = self.args.queries_num
        self.c_pool = c_pool
        self.valid_queries = self.c_pool.valid_queries
        self.specific = self.args.specific
        self.test_dataset = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')
        self.save_dir = args.model_save_path
        self.init_dataset_type()

    def train_query_encoder(self, epoch_num):
        self.c_pool.get_query_presentation()
        self.labels = torch.tensor(self.c_pool.query_presentation).float().to(self.device)

        self.init_query_encoder()
        self.best_epoch = 0
        self.best_loss = math.inf

        for epoch in range(epoch_num):
            batch = self.collector(self.plan_tree_dataset.collated_dicts)
            out = self.query_encoder(batch).to(self.device)
            self.val_loss = self.loss_func(out, self.labels)
            self.backward()

            if self.val_loss.item() < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = self.val_loss.item()
                self.save_model(self.workload)
            if epoch - self.best_epoch > 100:
                break
            logger1.info(f"epoch: {epoch}  loss: {self.val_loss.item()}")
        logger1.info(f"best epoch: {self.best_epoch}  best loss: {self.best_loss}")

    def eval_query_encoder(self, benchmark, epoch_num=1):

        start_time = time.time()
        self.c_pool.collect_data()
        self.c_pool.get_query_presentation()
        self.labels = torch.tensor(self.c_pool.query_presentation).float().to(self.device)

        self.init_query_encoder()
        self.best_epoch = 0
        self.best_loss = math.inf

        for epoch in range(epoch_num):
            batch = self.collector(self.plan_tree_dataset.collated_dicts)
            out = self.query_encoder(batch).to(self.device)
            self.val_loss = self.loss_func(out, self.labels)
            self.backward()

            if self.val_loss.item() < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = self.val_loss.item()
                self.save_model(self.workload)
            if epoch - self.best_epoch > 100:
                break
            logger1.info(f"epoch: {epoch}  loss: {self.val_loss.item()}")
        logger1.info(f"best epoch: {self.best_epoch}  best loss: {self.best_loss}")

        end_time = time.time()
        logger1.info(f'Time used was: {round((end_time - start_time)* 1000)}ms, '
                    f'{round(end_time - start_time) / 60}min, '
                    f'{round(end_time - start_time) / 3600}h')
        logger1.info(f"best epoch: {self.best_epoch}  best loss: {self.best_loss}")

    def get_query_encoding(self, benchmarks):
        batch = self.collector(self.plan_tree_dataset.collated_dicts)
        self.optimizer.zero_grad()
        out = self.query_encoder(batch, False).to(self.device)
        return out.cpu().detach().numpy()

    def backward(self):
        self.optimizer.zero_grad()
        self.last_val_loss = self.val_loss.item()
        self.val_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def collector(self, small_set):
        xs = [s['x'] for s in small_set]

        x = torch.cat(xs)
        attn_bias = torch.cat([s['attn_bias'] for s in small_set])
        rel_pos = torch.cat([s['rel_pos'] for s in small_set])
        heights = torch.cat([s['heights'] for s in small_set])

        return Batch(attn_bias, rel_pos, heights, x)

    def init_dataset_type(self):
        self.plan_tree_dataset = PlanTreeDataset(self.args, self.valid_queries)

    def init_query_encoder(self):
        self.query_encoder = QueryFormer(emb_size=64, ffn_dim=128, head_size=12, dropout=0.1, n_layers=8,
                                         use_sample=False, use_hist=False, pred_hid=256)

        # best params
        # 5e-4 200 0.8
        self.optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=self.lr)  # 1e-3
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size,
                                             gamma=self.gamma)  # 0.9

        logger1.info(f"lr={self.lr}  step_size={self.step_size}  gamma={self.gamma}")

    def save_model(self, model_name=''):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        with open(os.path.join(self.save_dir, 'query_encoder','query_encoder_queryformer_' + model_name + '.pickle'), 'wb') as file_model:
            pickle.dump(self.query_encoder, file_model)

    def load_model(self, model_name=''):
        with open(os.path.join(self.save_dir,'query_encoder', 'query_encoder_queryformer_' + model_name + '.pickle'), 'rb') as f:
            self.query_encoder = pickle.load(f)

        # best params
        # 5e-4 200 0.8
        self.optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=self.lr)  # 1e-3
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size,
                                             gamma=self.gamma)  # 0.9

        logger1.info(f"lr={self.lr}  step_size={self.step_size}  gamma={self.gamma}")
