import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
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


class FeatureEmbed(nn.Module):
    # def __init__(self, embed_size=32, types=32, columns=300):
    def __init__(self, embed_size=256*10, types=32, columns=300):
        super(FeatureEmbed, self).__init__()
        # type | table | column*2 | join table*3, join columns *3| cost | cardinality | index column*2
        # feature组成：node type、涉及到的table、column*2、join的table(固定占3格)、join columns *3、cost、cardinality、index column*2
        # node type、index的变化进行embedding，join的table(固定占3格)、涉及到的table直接加上来，cost、cardinality保留原数值
        self.embed_size = embed_size
        self.typeEmbed = nn.Embedding(types, embed_size, padding_idx=0)
        # self.indexEmbed = nn.Embedding(index_change, embed_size, padding_idx=0)
        self.columnEmbed = nn.Embedding(columns, embed_size, padding_idx=0)

        # self.project = nn.Linear(embed_size * 12 + 1 + 1, embed_size * 12 + 1 + 1)
        self.project = nn.Linear(embed_size, embed_size)

    def forward(self, feature):
        final = F.leaky_relu(self.project(feature))
        return final

    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())
        return emb.squeeze(2)

    def getcolumn(self, columnId):
        emb = self.columnEmbed(columnId.long())
        return emb

    def get_output_size(self):
        size = self.embed_size
        return size


class TableEmbed(nn.Module):
    def __init__(self, embed_size=32, output_dim=32, n_tables=32):
        super(TableEmbed, self).__init__()
        # 最多有n_tables张表
        self.tableEmbed = nn.Embedding(n_tables, embed_size)
        self.linearTable = nn.Linear(embed_size, output_dim)

    def forward(self, feature):
        output = self.tableEmbed(feature)
        tables_embedding = F.leaky_relu(self.linearTable(output))
        return tables_embedding


class QueryFormer(nn.Module):
    def __init__(self, emb_size=32, ffn_dim=32, head_size=8, dropout=0.1, attention_dropout_rate=0.1, n_layers=8,
                 use_sample=True, use_hist=True, bin_number=50, pred_hid=22):

        super(QueryFormer, self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size // 8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size // 8 + 1
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)

        # self.embbed_layer = FeatureEmbed(emb_size)
        self.embbed_layer = FeatureEmbed()

        self.pred = Prediction(hidden_dim, pred_hid)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)

    def forward(self, batched_data, pred=True):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x

        heights = batched_data.heights

        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1,
                                                             2)  # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        # x_view = x.view(-1, 1165)
        x_view = x.view(-1, 256*10)
        node_feature = self.embbed_layer(x_view).view(n_batch, -1, self.hidden_dim)

        # -1 is number of dummy
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        if pred:
            return self.pred(output[:, 0, :])
        else:
            return output[:, 0, :]

class QueryFormerOld(nn.Module):
    def __init__(self, emb_size=32, ffn_dim=32, head_size=8, dropout=0.1, attention_dropout_rate=0.1, n_layers=8,
                 use_sample=True, use_hist=True, bin_number=50, pred_hid=256):

        super(QueryFormerOld, self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size // 8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size // 8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)

        self.embbed_layer = FeatureEmbed(emb_size, use_sample=use_sample, use_hist=use_hist, bin_number=bin_number)

        self.pred = Prediction(hidden_dim, pred_hid)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)

    def forward(self, batched_data):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x

        heights = batched_data.heights

        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1,
                                                             2)  # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch, -1, self.hidden_dim)

        # -1 is number of dummy

        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        # return self.pred(output[:, 0, :]), self.pred2(output[:, 0, :])
        return self.pred(output[:, 0, :])


class PlanChanger(nn.Module):
    def __init__(self, emb_size=32, ffn_dim=32, head_size=8, join_schema_head_size=2, dropout=0.1,
                 attention_dropout_rate=0.1, n_layers=8, join_schema_layers=2, pred_hid=256
                 ):

        super(PlanChanger, self).__init__()
        hidden_dim = emb_size * 12 + 1 + 1
        self.join_schema_output_dim = emb_size

        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.join_schema_head_size = join_schema_head_size

        # 当输入类别值为0时，被识别为pad id，输出值为0
        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.tree_bias_encoder = nn.Embedding(64, head_size, padding_idx=0)
        self.join_bias_encoder = nn.Embedding(8, join_schema_head_size, padding_idx=0)

        # self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]

        join_schema_encoders = [
            EncoderLayer(self.join_schema_output_dim, ffn_dim, dropout, attention_dropout_rate, join_schema_head_size)
            for _ in range(join_schema_layers)]

        self.layers = nn.ModuleList(encoders)
        self.join_schema_layers = nn.ModuleList(join_schema_encoders)

        # 也就是只对node 内部的hidden_dim长的向量，做了归一化。即每个node单独归一化
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)

        self.embbed_layer = FeatureEmbed(emb_size)
        self.join_schema_embbed_layer = TableEmbed(output_dim=self.join_schema_output_dim)
        # 分别是输入维数和 最后的预测网络中间层神经元的个数
        self.pred = Prediction(hidden_dim, pred_hid)

    def forward(self, batched_data):
        features = batched_data.features
        attention_bias = batched_data.attention_bias

        attention_bias_SuperNode = batched_data.attention_bias_SuperNode.clone()
        join_schema_bias = batched_data.join_schema_bias
        join_features = batched_data.join_features
        heights = batched_data.heights

        n_batch = features.shape[0]
        n_table = join_schema_bias.shape[1]
        assert n_table == join_features.shape[0]
        # 先对join schema进行处理

        # join schema 的 Bias处理
        # join_schema_bias=join_schema_bias.unsqueeze(1).repeat(1, self.join_schema_head_size, 1, 1)
        join_schema_bias = self.join_bias_encoder(join_schema_bias.long())
        # print(join_schema_bias.shape)
        join_schema_bias = join_schema_bias.permute(0, 3, 1,
                                                    2)  # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        # print(join_schema_bias.shape)
        # 将表的编号进行encoder

        # 所有的表的feature，暂时将其扩充到batch大小了，后续为了效率可以试试，join_features、join_schema_bias不复制，毕竟对所有的都一样
        join_features = join_features.unsqueeze(1).repeat(n_batch, 1).view(-1, n_table)
        join_features = self.join_schema_embbed_layer(join_features.long()).view(n_batch, -1,
                                                                                 self.join_schema_output_dim)

        for enc_layer in self.join_schema_layers:
            join_features = enc_layer(join_features, join_schema_bias)

        # 处理Node features
        attention_bias_SuperNode = attention_bias_SuperNode.unsqueeze(0).repeat(n_batch, 1, 1)
        attention_bias_SuperNode = attention_bias_SuperNode.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        attention_bias = self.tree_bias_encoder(attention_bias.long()).permute(0, 3, 1,
                                                                               2)  # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        attention_bias_SuperNode[:, :, 1:, 1:] = attention_bias_SuperNode[:, :, 1:, 1:] + attention_bias

        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        attention_bias_SuperNode[:, :, 1:, 0] = attention_bias_SuperNode[:, :, 1:, 0] + t
        attention_bias_SuperNode[:, :, 0, :] = attention_bias_SuperNode[:, :, 0, :] + t

        node_feature = self.embbed_layer(features, join_features)
        node_feature = node_feature + self.height_encoder(heights.long())

        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, attention_bias_SuperNode)
        output = self.final_ln(output)

        return self.pred(output[:, 0, :])

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        # 输入为(batch size, seq size即序列长度 ,输入向量维度即hidden_size)，相当于对每个单词自己，通过两层神经元变换
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()
        # 注意力头数
        self.head_size = head_size

        # hidden_size是输入seq的中，每个词的维度。

        # att_size是d_k（d_q的长度一定和d_k一致）、d_v的长度，也就是每个注意力头下的q_i k_i v_i的输出长度，
        # 通常att_size == hidden_size // head_size，即qkv三个矩阵大小在不同的heads下是几乎相同的（有时候会根据取整的原因，差一点）
        self.att_size = att_size = hidden_size // head_size
        # 在queryFormer中写了，是 根号下dk ，用来帮助梯度下降时更稳定的
        self.scale = att_size ** -0.5

        # 是所有头的qkv矩阵
        # 其中lin=nn.Linear(输入向量维度，输出向量维度)，而输入到lin的可以是任意维度的，比如这里(batch size, seq size即序列长度 ,输入向量维度即hidden_size)
        # 会以最后一个维度的向量作为输出，进行线性变换，input*weight+bias，输出(batch size, seq size即序列长度 ,输出向量维度即 head_size * att_size)
        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        # 在训练时会随机的丢弃一些，但是测试时不会
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        '''
        在这个代码中，输入张量的每个维度的含义如下：

        q: (batch_size, q_len, hidden_size)，表示 batch_size 个查询向量，每个向量长度为 hidden_size, 总共有 q_len 个查询向量。
        k: (batch_size, k_len, hidden_size)，表示 batch_size 个键向量，每个向量长度为 hidden_size, 总共有 k_len 个键向量。
        v: (batch_size, v_len, hidden_size)，表示 batch_size 个值向量，每个向量长度为 hidden_size, 总共有 v_len 个值向量。
        attn_bias: (batch_size, 1, q_len, k_len)，表示 batch_size 个注意力偏置项，每个偏置项都是一个 q_len x k_len 的矩阵，这个矩阵中的每个元素都是一个偏置项。
        
        '''
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
