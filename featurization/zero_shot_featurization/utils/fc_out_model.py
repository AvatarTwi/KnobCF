import math

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from featurization.zero_shot_featurization.utils import losses, activations


class DynamicLayer(nn.Module):
    """
    A layer of an MLP with a specifiable dropout, norm and activation class
    """

    def __init__(self, p_dropout=None, activation_class_name=None, activation_class_kwargs=None, norm_class_name=None,
                 norm_class_kwargs=None, inplace=False, **kwargs):
        super().__init__()
        # initialize base NN
        self.inplace = inplace
        self.p_dropout = p_dropout
        self.act_class = activations.__dict__[activation_class_name]
        self.act_class_kwargs = activation_class_kwargs
        self.norm_class = nn.__dict__[norm_class_name]
        self.norm_class_kwargs = norm_class_kwargs

    def get_act(self):
        # return self.act_class(inplace=self.inplace, **self.act_class_kwargs)
        return self.act_class(**self.act_class_kwargs)

    def get_norm(self, num_feats):
        # return self.norm_class(num_feats, inplace=self.inplace, **self.norm_class_kwargs)
        return self.norm_class(num_feats, **self.norm_class_kwargs)


# Residual block
class ResidualBlock(DynamicLayer):
    """
    One residual layer in an MLP.
    """

    def __init__(self, layer_in, layer_out, norm=False, activation=False, dropout=False, final_out_layer=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert layer_out == layer_in
        hidden_dim = layer_in

        self.activation = activation
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        if norm:
            self.layers.append(self.get_norm(hidden_dim))
        if activation:
            self.layers.append(self.get_act())

        # if dropout:
        #     self.layers.append(nn.Dropout(self.p_dropout))
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        if norm:
            self.layers.append(self.get_norm(hidden_dim))
        if dropout and not final_out_layer:
            # this can never be run inplace since it raises an error
            self.layers.append(nn.Dropout(self.p_dropout, inplace=False))

        if activation:
            self.final_act_layer = self.get_act()

    def forward(self, x):
        residual = x
        out = x
        for l in self.layers:
            out = l(out)

        out += residual
        if self.activation:
            out = self.final_act_layer(out)

        return out


class FcLayer(DynamicLayer):
    """
    One layer in an MLP.
    """

    def __init__(self, layer_in, layer_out, norm=False, activation=False, dropout=False, final_out_layer=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_in = layer_in

        layers = []
        layers.append(nn.Linear(layer_in, layer_out))
        if activation:
            layers.append(self.get_act())
        if norm:
            layers.append(self.get_norm(layer_out))
        if dropout and not final_out_layer:
            # this can never be run inplace since it raises an error
            layers.append(nn.Dropout(self.p_dropout, inplace=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        # print(self.layer_in)
        return self.layers(x)


class FcOutModel(DynamicLayer):
    """
    Model with an MLP (Multi-Layer-Perceptron) on top.
    """

    def __init__(self, output_dim=None, input_dim=None,hidden=None, n_layers=None, width_factor=None,
                 residual=True, loss_class_name=None, loss_class_kwargs=None, task=None, **kwargs):
        super().__init__(**kwargs)

        self.task = task
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.no_input_required = input_dim == 0
        if hidden is not None:
            self.hidden_dim = hidden
        else:
            self.hidden_dim = self.input_dim

        if self.input_dim == 0:
            self.replacement_param = Parameter(torch.Tensor(self.output_dim))
            # init.kaiming_uniform_(self.replacement_param, a=math.sqrt(5))
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.replacement_param)
            bound = 1 / math.sqrt(self.output_dim)
            init.uniform_(self.replacement_param, -bound, bound)
            return

        if loss_class_name is not None:
            self.loss_fxn = losses.__dict__[loss_class_name](self, **loss_class_kwargs)

        layer_dims = [self.input_dim] + [int(width_factor * self.hidden_dim)] * n_layers + [self.output_dim]

        layers = []
        for layer_in, layer_out in zip(layer_dims, layer_dims[1:]):
            if not residual or layer_in != layer_out:
                layers.append(FcLayer(layer_in, layer_out, **kwargs))
            else:
                layers.append(ResidualBlock(layer_in, layer_out, **kwargs))
        self.fcout = nn.Sequential(*layers)

import torch.nn.functional as F

# 作为整个网络最后，用来输出预测值
class Prediction(nn.Module):
    def __init__(self, in_feature=69, hid_units=256, contract=1, mid_layers=True, res_con=True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con

        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, hid_units // contract)
        self.mid_mlp2 = nn.Linear(hid_units // contract, hid_units)
        self.mid_mlp3 = nn.Linear(hid_units, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 22)

    def forward(self, features):

        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            mid = F.relu(self.mid_mlp3(mid))
            # 残差连接 residual connections
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out
