import math
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from config import config
from dataset.job.job_utils_serialize import JOBSerilalDataSet
from dataset.tpcc.tpcc_utils_serialize import TPCCSerilalDataSet
from dataset.tpch.tpch_utils_serialize import TPCHSerilalDataSet
from model.query_encoder_models.query_encoder import QueryEncoder

DATASET_TYPE = {
    "tpch": TPCHSerilalDataSet,
    "tpcc": TPCCSerilalDataSet,
    "job": JOBSerilalDataSet,
    # "ycsb": PSQLYCSB,
}


# Define model architecture

class MSCN(nn.Module):
    def __init__(self, table_feats, predicate_feats, hid_units):
        super(MSCN, self).__init__()
        self.table_feats = table_feats
        self.predicate_feats = predicate_feats
        self.test = True

        self.table_mlp1 = nn.Linear(table_feats, hid_units)
        self.table_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 2, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 22)
        self.input = None

    def update_input(self, input):
        self.input = input

    def forward(self):
        # tables, predicates, table_mask, predicate_mask = train[0], train[1], train[2], train[3]
        tables, predicates, table_mask, predicate_mask = self.input[:, :, :self.table_feats], \
                                                         self.input[:, :,
                                                         self.table_feats:self.table_feats + self.predicate_feats], \
                                                         self.input[:, :, -2].reshape(-1, 1, 1), \
                                                         self.input[:, :, -1].reshape(-1, 1, 1)

        hid_table = F.relu(self.table_mlp1(tables))
        hid_table = F.relu(self.table_mlp2(hid_table))
        hid_table = hid_table * table_mask  # Mask
        hid_table = torch.sum(hid_table, dim=1, keepdim=False)
        table_norm = table_mask.sum(1, keepdim=False)
        hid_table = hid_table / table_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid = torch.cat((hid_table, hid_predicate), 1)
        out = F.relu(self.out_mlp1(hid))
        if self.test:
            out = torch.sigmoid(self.out_mlp2(out))
        return out


class MSCNModel(QueryEncoder):
    def __init__(self, args, c_pool):
        super().__init__(args, c_pool)
        self.test_dataset = None
        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.save_dir = args.model_save_path

        # Initialize the neural units
        self.init_dataset_type()
        self.model = MSCN(self.dim_dict["table_list_len"], self.dim_dict["maxlen_plan"], 256).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)  # opt.lr
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.99)

        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.total_loss = None
        self._test_losses = dict()

    def init_dataset_type(self):
        dataset_type = DATASET_TYPE[config['benchmark_info']['workload']](args=self.args, c_pool=self.c_pool)
        self.input = dataset_type.dataset
        self.dim_dict = dataset_type.dim_dict

    def backward(self, epoch):
        self.last_total_loss = self.loss.item()
        if self.best > self.loss.item():
            self.best_epoch = epoch
            self.best = self.loss.item()
            self.save_model(config['benchmark_info']['workload'])
        self.loss.backward()

    def train_query_encoder(self, num_epochs):
        self.best = math.inf
        self.best_epoch = 0
        self.c_pool.get_query_presentation()
        self.label = torch.tensor(self.c_pool.query_presentation).float().to(self.device)
        self.model.update_input(self.input)

        for epoch in range(num_epochs):
            self.total_loss = 0.

            self.optimizer.zero_grad()

            outputs = self.model()

            self.loss = self.loss_func(outputs, self.label)

            self.backward(epoch)
            self.optimizer.step()
            self.scheduler.step()
            print("Epoch: {}, Loss: {}".format(epoch, self.loss.item()))

        print("Best epoch: {}, Best loss: {}".format(self.best_epoch, self.best))

    def get_query_encoding(self, benchmarks=None):
        self.model.test = False
        outputs = self.model()
        self.c_pool.query_encoding = outputs.cpu().detach().numpy()
        return self.c_pool.query_encoding

    def load_model(self, model_name):
        save_filename = 'query_encoder_%s_%s.pth' % ('mscn', model_name)
        save_path = os.path.join(self.save_dir, save_filename)

        with open(save_path, 'rb') as file_model:
            self.model = pickle.load(file_model)

    def save_model(self, model_name):
        save_filename = 'query_encoder_%s_%s.pth' % ('mscn', model_name)
        save_path = os.path.join(self.save_dir, save_filename)

        with open(save_path, 'wb') as file_model:
            pickle.dump(self.model, file_model)
