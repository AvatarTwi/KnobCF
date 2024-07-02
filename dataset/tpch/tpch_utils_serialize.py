import numpy as np
import collections, os

import torch
from sklearn.preprocessing import Normalizer
from torch.autograd import Variable

from dataset.tpch.attr_rel_dict import *
import pickle

from dataset.load_json_plan import load2json_plan

num_rel = 8
max_num_attr = 16
num_index = 23
SCALE = 1

op_data_dic = {}

TRAIN_TEST_SPLIT = 0.8

with open('dataset/tpch/attr_val_dict.pickle', 'rb') as f:
    attr_val_dict = pickle.load(f)

normalizer = Normalizer(norm='l2')


# need to normalize Plan Width, Plan Rows, Total Cost, Hash Bucket
def get_basics(plan_dict):
    return [plan_dict['Plan Width'], plan_dict['Plan Rows'], plan_dict['Total Cost']]


def get_rel_one_hot(rel_name):
    arr = [0] * num_rel
    arr[rel_names.index(rel_name)] = 1
    return arr


def get_index_one_hot(index_name):
    arr = [0] * num_index
    arr[index_names.index(index_name)] = 1
    return arr


def get_rel_attr_one_hot(rel_name, filter_line):
    attr_list = rel_attr_list_dict[rel_name]

    med_vec, min_vec, max_vec = [0] * max_num_attr, [0] * max_num_attr, [0] * max_num_attr

    for idx, attr in enumerate(attr_list):
        if attr in filter_line:
            med_vec[idx] = attr_val_dict['med'][rel_name][idx]
            min_vec[idx] = attr_val_dict['min'][rel_name][idx]
            max_vec[idx] = attr_val_dict['max'][rel_name][idx]
    return min_vec + med_vec + max_vec


def get_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Filter'])
    except:
        try:
            rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                                plan_dict['Recheck Cond'])
        except:
            if 'Filter' in plan_dict:
                print('************************* default *************************')
                print(plan_dict)
            rel_attr_vec = [0] * max_num_attr * 3

    return get_basics(plan_dict) + rel_vec + rel_attr_vec


def get_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Index Scan'

    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Index Cond'])
    except:
        if 'Index Cond' in plan_dict:
            print('********************* default rel_attr_vec *********************')
            print(plan_dict)
        rel_attr_vec = [0] * max_num_attr * 3

    res = get_basics(plan_dict) + rel_vec + rel_attr_vec + index_vec \
          + [1 if plan_dict['Scan Direction'] == 'Forward' else 0]

    return res


def get_bitmap_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Bitmap Index Scan'
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    return get_basics(plan_dict) + index_vec


def get_hash_input(plan_dict):
    return get_basics(plan_dict) + [plan_dict['Hash Buckets'] if 'Hash Buckets' in plan_dict.keys() else 1]


def get_join_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1
    par_rel_vec = [0] * len(parent_rel_types)
    if 'Parent Relationship' in plan_dict:
        par_rel_vec[parent_rel_types.index(plan_dict['Parent Relationship'].lower())] = 1
    return get_basics(plan_dict) + type_vec + par_rel_vec


def get_nested_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1

    return get_basics(plan_dict) + type_vec


def get_sort_key_input(plan_dict):
    kys = plan_dict['Sort Key']
    one_hot = [0] * (num_rel * max_num_attr)
    for key in kys:
        key = key.replace('(', ' ').replace(')', ' ')
        for subkey in key.split(" "):
            if subkey != ' ' and '.' in subkey:
                rel_name, attr_name = subkey.split(' ')[0].split('.')
                if rel_name in rel_names:
                    one_hot[rel_names.index(rel_name) * max_num_attr
                            + rel_attr_list_dict[rel_name].index(attr_name.lower())] = 1

    return one_hot


def get_sort_input(plan_dict):
    sort_meth = [0] * len(sort_algos)
    if 'Sort Method' in plan_dict:
        if "external" not in plan_dict['Sort Method'].lower():
            sort_meth[sort_algos.index(plan_dict['Sort Method'].lower())] = 1

    return get_basics(plan_dict) + get_sort_key_input(plan_dict) + sort_meth


def get_aggreg_input(plan_dict):
    strat_vec = [0] * len(aggreg_strats)
    strat_vec[aggreg_strats.index(plan_dict['Strategy'].lower())] = 1
    partial_mode_vec = [0] if plan_dict['Parallel Aware'] == 'false' else [1]
    return get_basics(plan_dict) + strat_vec + partial_mode_vec


def get_modify_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])

    return get_basics(plan_dict) + rel_vec


TPCH_GET_INPUT = \
    {
        "Hash Join": get_join_input,
        "Merge Join": get_join_input,
        "Nested Loop": get_nested_input,
        "Seq Scan": get_scan_input,
        "Index Scan": get_index_scan_input,
        "Index Only Scan": get_index_scan_input,
        "Bitmap Heap Scan": get_scan_input,
        "Bitmap Index Scan": get_bitmap_index_scan_input,
        "Sort": get_sort_input,
        "Hash": get_hash_input,
        "Aggregate": get_aggreg_input,
        "ModifyTable": get_modify_input,
    }

norm_model = Normalizer(norm='l2')
TPCH_GET_INPUT = collections.defaultdict(lambda: get_basics, TPCH_GET_INPUT)


###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################

class TPCHSerilalDataSet():
    def __init__(self, args, c_pool):
        """
            Initialize the dataset by parsing the data files.
            Perform train test split and normalize each feature using mean and max of the train dataset.

            self.dataset is the train dataset
            self.test_dataset is the test dataset
        """
        self.args = args
        self.input_func = TPCH_GET_INPUT
        path = os.path.join(self.args.local_datadir + 'tpch4base' + '/', 'serverlog0.txt')
        all_groups = load2json_plan(path)[:self.args.queries_num]
        all_groups = [grp for i,grp in enumerate(all_groups) if i in c_pool.valid_queries]

        self.table_list = []
        self.join_onehot = []
        self.source_list = []
        self.total_time = []
        self.total_cost = []

        for grp in all_groups:
            if grp == []:
                continue
            new_samp_dict = self.get_input(grp)
            self.join_onehot.extend([new_samp_dict["join_onehot"] for i in range(len(new_samp_dict["feat_vec"]))])
            self.source_list.extend(new_samp_dict["feat_vec"])
            self.total_time.extend(new_samp_dict["total_time"])
            self.total_cost.extend(new_samp_dict["total_cost"])

        self.align()
        self.total_time, min_val, max_val = self.normalize_labels(self.total_time)

        self.dataset = self.make_dataset(self.join_onehot, self.source_list, self.total_time)

        self.dim_dict = {
            "min_val": min_val,
            "max_val": max_val,
            "table_list_len": len(rel_names),
            "maxlen_plan": len(self.source_list[0])
        }

    def align_table(self):
        max_len = max([len(list) for list in self.table_list])
        for idx,table in enumerate(self.table_list):
            self.table_list[idx] = np.vstack((table,np.zeros((max_len - len(table),len(all_dicts)))))

    def align(self):
        max_len = max([source.shape[1] for source in self.source_list])
        for idx,source in enumerate(self.source_list):
            self.source_list[idx] = np.concatenate((source.flatten(), np.zeros(max_len - source.shape[1])), axis=0)

    def normalize_labels(self, labels):
        labels = np.array([np.log(float(l)) for l in labels if l != 0])
        min_val = labels.min()
        max_val = labels.max()
        labels_norm = (labels - min_val) / (max_val - min_val)
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0)
        return labels_norm, min_val, max_val

    def make_dataset(self, join_onehots, predicates, labels):
        join_onehot_masks = []
        join_onehot_tensors = []
        for join_onehot in join_onehots:
            join_onehot_tensor = np.vstack([join_onehot])
            join_onehot_tensor = np.array(join_onehot_tensor)
            join_onehot_mask = np.ones_like(join_onehot_tensor).mean(1, keepdims=True)
            join_onehot_tensors.append(np.expand_dims(join_onehot_tensor, 0))
            join_onehot_masks.append(np.expand_dims(join_onehot_mask, 0))

        join_onehot_tensors = np.vstack(join_onehot_tensors)
        join_onehot_tensors = torch.FloatTensor(join_onehot_tensors)
        join_onehot_masks = np.vstack(join_onehot_masks)
        join_onehot_masks = torch.FloatTensor(join_onehot_masks)

        predicate_masks = []
        predicate_tensors = []
        for predicate in predicates:
            predicate_tensor = np.vstack([predicate])
            predicate_tensor = np.array(predicate_tensor)
            predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
            predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
            predicate_masks.append(np.expand_dims(predicate_mask, 0))

        predicate_tensors = np.vstack(predicate_tensors)
        predicate_tensors = torch.FloatTensor(predicate_tensors)
        predicate_masks = np.vstack(predicate_masks)
        predicate_masks = torch.FloatTensor(predicate_masks)

        target_tensor = torch.FloatTensor(labels)

        join_onehots, predicates, targets = join_onehot_tensors.cuda(), predicate_tensors.cuda(), target_tensor.cuda()
        join_onehot_masks, predicate_masks = join_onehot_masks.cuda(), predicate_masks.cuda()

        join_onehot_tensors, predicate_tensors, target_tensor = Variable(join_onehots), Variable(predicates), Variable(
            targets)
        join_onehot_masks, predicate_masks = Variable(join_onehot_masks), Variable(predicate_masks)

        trainX = torch.concatenate((join_onehot_tensors, predicate_tensors, join_onehot_masks, predicate_masks),dim=2)

        return trainX

    def grouping(self, data):
        """
            Groups the queries by their query plan structure

            Args:
            - data: a list of dictionaries, each being a query from the dataset

            Returns:
            - enum    : a list of same length as data, containing the group indexes for each query in data
            - counter : number of distinct groups/templates
        """

        def hash(plan_dict):
            res = plan_dict['Node Type']
            if 'Plans' in plan_dict:
                for chld in plan_dict['Plans']:
                    res += hash(chld)
            return res

        counter = 0
        string_hash = []
        enum = []
        for plan_dict in data:
            string = hash(plan_dict)
            try:
                idx = string_hash.index(string)
                enum.append(idx)
            except:
                idx = counter
                counter += 1
                enum.append(idx)
                string_hash.append(string)

        assert (counter > 0)
        return enum, counter

    def get_input(self, data, first=True):  # Helper for sample_data
        """
            Vectorize the input of a list of queries that have the same plan structure (of the same template/group)

            Args:
            - data: a list of plan_dict, each plan_dict correspond to a query plan in the dataset;
                    requires that all plan_dicts is of the same query template/group

            Returns:
            - new_samp_dict: a dictionary, where each level has the following attribute:
                -- node_type     : name of the operator
                -- subbatch_size : number of queries in data
                -- feat_vec      : a numpy array of shape (batch_size x feat_dim) that's
                                   the vectorized inputs for all queries in data
                -- children_plan : list of dictionaries with each being an output of
                                   a recursive call to get_input on a child of current node
                -- total_time    : a vector of prediction target for each query in data
                -- is_subplan    : if the queries are subplans
        """

        def get_op_one_hot(op):
            arr = [0] * len(all_dicts)
            arr[all_dicts.index(op)] = 1
            return np.array(arr)

        # Merge Cond | Hash Cond
        def get_join_one_hot(cond):
            arr = [0] * len(rel_names)
            for idx, rel in enumerate(rel_names):
                if rel in cond:
                    arr[idx] = 1
            return np.array(arr)

        def join_combine(array1, array2):
            for idx, arr1 in enumerate(array1):
                if arr1 == 1:
                    array2[idx] = 1
            return array2

        new_samp_dict = {}
        new_samp_dict["node_type"] = get_op_one_hot(data["Node Type"])

        if data["Node Type"] == 'Hash Join':
            new_samp_dict["join_onehot"] = get_join_one_hot(data["Hash Cond"])
        elif data["Node Type"] == 'Merge Join':
            new_samp_dict["join_onehot"] = get_join_one_hot(data["Merge Cond"])
        else:
            new_samp_dict["join_onehot"] = np.array([0] * len(rel_names))

        new_samp_dict["subbatch_size"] = len(data)

        feat_vec = np.array(self.input_func[data["Node Type"]](data)).reshape(1, -1)

        # normalize feat_vec
        # feat_vec = norm_model.fit_transform(feat_vec)

        feat_vec = np.array(
            [np.hstack(
                (new_samp_dict["node_type"], feat))
                for feat in feat_vec])

        total_time = data['Actual Total Time']
        total_cost = data['Total Cost']

        if 'Plans' in data:
            for i in range(len(data['Plans'])):
                child_plan_dict = self.get_input(data['Plans'][i], False)
                new_samp_dict["node_type"] = np.vstack((child_plan_dict["node_type"], new_samp_dict["node_type"]))
                new_samp_dict["join_onehot"] = join_combine(child_plan_dict["join_onehot"],
                                                            new_samp_dict["join_onehot"])
                feat_vec = np.hstack((child_plan_dict["feat_vec"], feat_vec))

        if first:
            new_samp_dict["feat_vec"] = np.array([feat_vec]).astype(np.float32)
            new_samp_dict["total_time"] = np.array([total_time]).astype(np.float32)
            new_samp_dict["total_cost"] = np.array([total_cost]).astype(np.float32)
        else:
            new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
            new_samp_dict["total_time"] = np.array(total_time).astype(np.float32)
            new_samp_dict["total_cost"] = np.array(total_cost).astype(np.float32)

        if 'Subplan Name' in data:
            new_samp_dict['is_subplan'] = True
        else:
            new_samp_dict['is_subplan'] = False

        return new_samp_dict
