import os
from collections import deque

import numpy as np
from torch.utils.data import Dataset

from dataset.load_json_plan import load2json_plan
from model.query_encoder_models.QueryFormerModule.benchmark_utils import BENCHMARK_UTILS
from model.database_util import *


class PlanTreeDataset(Dataset):
    def __init__(self,args, valid_queries):
        self.args = args
        self.workload = self.args.workload
        path = os.path.join(self.args.local_datadir + self.workload + '4base' + '/', 'serverlog0.txt')
        nodes = load2json_plan(path)
        nodes = [node for idx,node in enumerate(nodes) if idx in valid_queries]
        self.length = len(nodes)
        self.input_func = BENCHMARK_UTILS[self.workload]
        idxs = list([i for i in range(len(nodes))])

        self.treeNodes = []  ## for mem collection
        self.collated_dicts = [self.js_node2dict(i, node) for i,node in zip(idxs,nodes)]

    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)

        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def traversePlan(self, plan, idx):  # bfs accumulate plan
        root = TreeNode()
        root.feature = np.array(self.input_func[plan["Node Type"]](plan)).reshape(1, -1)
        root.query_id = idx

        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx)
                node.parent = root
                root.addChild(node)
        return root

    def node2dict(self, treeNode):
        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),
        }

    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node=30, rel_pos_max=20):
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True

            shortest_path_result = floyd_warshall_rewrite(adj.numpy())

        rel_pos = torch.from_numpy((shortest_path_result)).long()

        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')

        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)

        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }

    def topo_sort(self, root_node):
        adj_list = []  # from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            feature = node.feature.flatten()
            new_x = np.zeros([256], dtype=feature[0].dtype)
            new_x[:len(feature)] = feature
            features.append(new_x)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1

        return adj_list, num_child, features

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.collated_dicts[idx]
