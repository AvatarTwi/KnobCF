import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from config import config
from dataset.job.job_utils import job_dim_dict
from dataset.tpcc.tpcc_utils import TPCCDataset, tpcc_dim_dict
from dataset.tpch.tpch_utils import TPCHDataSet, tpch_dim_dict
from model.query_encoder_models.query_encoder import QueryEncoder

DATASET_TYPE = {
    "tpch": TPCHDataSet,
    "tpcc": TPCCDataset,
    # "job": JOBDataset(args=self.args),
    # "ycsb": PSQLYCSB,
}

DIM_DICT = {
    "tpch": tpch_dim_dict,
    "tpcc": tpcc_dim_dict,
    "job": job_dim_dict,
}
basic = 3


# For computing loss
def squared_diff(output, target):
    return torch.sum((output - target) ** 2)


###############################################################################
#                        Operator Neural Unit Architecture                    #
###############################################################################
# Neural Unit that covers all operators
class NeuralUnit(nn.Module):
    """Define a Resnet block"""

    def __init__(self, node_type, dim_dict, num_layers=5, hidden_size=256,
                 output_size=22, norm_enabled=False):
        """
        Initialize the InternalUnit
        """
        super(NeuralUnit, self).__init__()
        self.node_type = node_type
        self.dense_block = self.build_block(num_layers, hidden_size, output_size,
                                            input_dim=dim_dict[node_type])

    def build_block(self, num_layers, hidden_size, output_size, input_dim):
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        assert (num_layers >= 2)
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for i in range(num_layers - 2):
            dense_block += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        dense_block += [nn.Linear(hidden_size, output_size), nn.ReLU()]

        for layer in dense_block:
            try:
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
        return nn.Sequential(*dense_block)

    def forward(self, x):
        """ Forward function """
        out = self.dense_block(x)
        return out

###############################################################################
#                               QPP Net Architecture                          #
###############################################################################

class QPPNet(QueryEncoder):
    def __init__(self, args, c_pool):
        super().__init__(args, c_pool)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu:0')

        self.last_total_loss = None
        self.last_pred_err = None
        self.pred_err = None
        self.rq = 0
        self.last_rq = 0

        self.loss_fn = squared_diff
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.total_loss = None
        self._test_losses = dict()
        self.best = math.inf
        self.init_qppnet_query_encoder()

    def init_qppnet_query_encoder(self):
        dataset_type = DATASET_TYPE[config['benchmark_info']['workload']](args=self.args, c_pool=self.c_pool)
        self.input = dataset_type.dataset
        self.dim_dict = DIM_DICT[config['benchmark_info']['workload']]

        # Initialize the neural units
        self.units = {}
        self.optimizers, self.schedulers = {}, {}

        for operator in self.dim_dict:
            self.units[operator] = NeuralUnit(operator, self.dim_dict).to(self.device)
            optimizer = torch.optim.Adam(self.units[operator].parameters(), lr=8e-4)  # opt.lr
            self.optimizers[operator] = optimizer
            self.schedulers[operator] = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.6)

        self.acc_loss = {operator: [self.dummy] for operator in self.dim_dict}
        self.curr_losses = {operator: 0 for operator in self.dim_dict}

    def _forward_oneQ_batch(self, samp_batch, last=False):
        '''
        Calcuates the loss for a batch of queries from one query template

        compute a dictionary of losses for each operator

        return output_vec, where 1st col is predicted time
        '''
        feat_vec = samp_batch['feat_vec']
        input_vec = torch.from_numpy(feat_vec).to(self.device)
        subplans_time = []
        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec, _ = self._forward_oneQ_batch(child_plan_dict,False)
            if not child_plan_dict['is_subplan']:
                input_vec = torch.cat((input_vec, child_output_vec), axis=1).to(self.device)
            else:
                subplans_time.append(torch.index_select(child_output_vec, 1, torch.zeros(1, dtype=torch.long)))

        expected_len = self.dim_dict[samp_batch['node_type']]
        if expected_len > input_vec.size()[1]:
            add_on = torch.zeros(input_vec.size()[0], expected_len - input_vec.size()[1]).to(self.device)
            input_vec = torch.cat((input_vec, add_on), axis=1)

        # print(samp_batch['node_type'], input_vec)
        output_vec = self.units[samp_batch['node_type']](input_vec)
        # print(output_vec.shape)
        pred_time = torch.index_select(output_vec, 1, torch.zeros(1, dtype=torch.long).to(self.device)).to(self.device)
        # pred_time assumed to be the first col

        cat_res = torch.cat([pred_time] + subplans_time, axis=1).to(self.device)
        pred_time = torch.sum(cat_res, 1)

        if last:
            loss = (self.loss_func(output_vec, self.temp_label)-torch.zeros(1).to(self.device)).to(self.device)
            self.acc_loss[samp_batch['node_type']].append(loss)
        else:
            loss = (pred_time - torch.from_numpy(samp_batch['total_time']).to(self.device)) ** 2
            self.acc_loss[samp_batch['node_type']].append(loss)

        try:
            assert (not (torch.isnan(output_vec).any()))
        except:
            print("feat_vec", feat_vec, "input_vec", input_vec)
            if torch.cuda.is_available():
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].module.cpu().state_dict())
            else:
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].cpu().state_dict())
            exit(-1)
        return output_vec, pred_time

    def _forward(self, epoch):
        # self.input is a list of preprocessed plan_vec_dict
        losses = torch.zeros(1).to(self.device)
        total_loss = torch.zeros(1).to(self.device)
        total_losses = {operator: [torch.zeros(1).to(self.device)] \
                        for operator in self.dim_dict}

        for idx, samp_dict in enumerate(self.input):
            # first clear prev computed losses
            del self.acc_loss
            self.acc_loss = {operator: [self.dummy] for operator in self.dim_dict}

            self.temp_label = self.label[idx]

            output_vec, pred_time = self._forward_oneQ_batch(samp_dict, True)
            loss = self.loss_func(output_vec, self.temp_label)
            losses += loss

            D_size = 0
            subbatch_loss = torch.zeros(1).to(self.device)

            for operator in self.acc_loss:
                all_loss = torch.cat(self.acc_loss[operator])
                D_size += all_loss.shape[0]
                subbatch_loss += torch.sum(all_loss)

                total_losses[operator].append(all_loss)
            subbatch_loss = torch.mean(torch.sqrt(subbatch_loss / D_size))
            total_loss += subbatch_loss * samp_dict['subbatch_size']

        self.curr_losses = {operator: torch.mean(torch.cat(total_losses[operator])).item() for operator in
                            self.dim_dict}
        self.total_loss = torch.mean(total_loss)
        print("epoch: ", epoch, "loss: ", loss)

    def backward(self, epoch):
        self.last_total_loss = self.total_loss.item()
        if self.best > self.total_loss.item():
            self.best = self.total_loss.item()
            self.save_model(config['benchmark_info']['workload'])
            self.best_epoch = epoch
        self.total_loss.backward()
        self.total_loss = None

    def train_query_encoder(self, epoch_num):
        self.c_pool.get_query_presentation()
        self.best_epoch = 0
        self.best_loss = math.inf
        self.label = torch.tensor(self.c_pool.query_presentation).float().to(self.device)

        for epoch in range(epoch_num):
            self._forward(epoch)
            for operator in self.optimizers:
                self.optimizers[operator].zero_grad()
            self.backward(epoch)

            for operator in self.optimizers:
                self.optimizers[operator].step()
                self.schedulers[operator].step()
            if epoch - self.best_epoch > 100:
                break

        self.load_model(config['benchmark_info']['workload'])
        self._forward(epoch_num)

    def get_query_encoding(self):
        outs = []
        for idx, samp_dict in enumerate(self.input):
            # first clear prev computed losses
            del self.acc_loss
            self.acc_loss = {operator: [self.dummy] for operator in self.dim_dict}

            out, _ = self._forward_oneQ_batch(samp_dict)
            outs.append(out.cpu().detach().numpy())

        self.c_pool.query_encoding = np.concatenate(outs, axis=0)
        print("query encoding shape: ", self.c_pool.query_encoding.shape)

    def save_model(self, model_name):
        for name, unit in self.units.items():
            save_filename = '%s_query_encoder_%s_%s.pth' % ('qppnet',model_name, name)
            save_path = os.path.join(self.save_dir, save_filename)

            if torch.cuda.is_available():
                torch.save(unit.state_dict(), save_path)
                unit.to(self.device)
            else:
                torch.save(unit.cpu().state_dict(), save_path)

    def load_model(self, model_name):
        for name in self.units:
            save_filename = '%s_query_encoder_%s_%s.pth' % ('qppnet',model_name, name)
            save_path = os.path.join(self.save_dir, save_filename)
            if not os.path.exists(save_path):
                raise ValueError("model {} doesn't exist".format(save_path))

            self.units[name].load_state_dict(torch.load(save_path))
