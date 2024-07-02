import collections
import json
import os
import pickle
import re

import numpy as np

from dataset.tpch.attr_rel_dict import *
from dataset.load_json_plan import load2json_plan
from sklearn.preprocessing import Normalizer

basics = 3  # get_basics(plan_dict)
num_rel = len(rel_names)
max_num_attr = 16
num_index = len(index_names)
SCALE = 1

knob_num = 3
op_data_dic = {}

TRAIN_TEST_SPLIT = 0.8

tpch_dim_dict = {'Seq Scan': num_rel + max_num_attr * 3 + basics,
                 'Index Scan': num_index + num_rel + max_num_attr * 3 + basics + 1,
                 'Index Only Scan': num_index + num_rel + max_num_attr * 3 + basics + 1,
                 'Bitmap Heap Scan': basics + num_rel + max_num_attr * 3 + 32,
                 'Bitmap Index Scan': num_index + basics,
                 'Sort': basics + len(sort_algos) + 128 + 32,
                 'Hash': basics + 1 + 32,
                 'Hash Join': basics + len(join_types) + len(parent_rel_types) + 32 * 2,
                 'Merge Join': basics + len(join_types) + len(parent_rel_types) + 32 * 2,
                 'Aggregate': basics + len(aggreg_strats) + 1 + 32,
                 'Nested Loop': 32 * 2 + len(join_types) + basics,
                 'Limit': 32 + basics,
                 'Subquery Scan': 32 + basics,
                 'Materialize': 32 + basics,
                 'Gather Merge': 32 + basics,
                 'Gather': 32 + basics,
                 'BitmapAnd': 32 * 2 + basics,
                 'Memoize': 32 + basics
                 }

with open('dataset/tpch/attr_val_dict.pickle', 'rb') as f:
    attr_val_dict = pickle.load(f)


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

class TPCHDataSet:
    def __init__(self, args):
        """
            Initialize the dataset by parsing the data files.
            Perform train test split and normalize each feature using mean and max of the train dataset.

            self.dataset is the train dataset
            self.test_dataset is the test dataset
        """
        self.args = args
        self.input_func = TPCH_GET_INPUT
        path = os.path.join(self.args.local_datadir + 'tpch4base' + '/',
                            'serverlog0.txt')
        explain_data = load2json_plan(path)[:self.args.queries_num]
        self.dataset = [self.get_input(grp) for grp in explain_data if grp != []]

    def get_all_plans(self, filepath):
        json_obj = []
        pattern_num = re.compile(r'\d+\.?\d*')

        with open(filepath, "r") as f:
            lines = [line for line in f.readlines()]
            lineid = 0
            while lineid < len(lines):
                if ' UTC [' not in lines[lineid] and 'CST [' not in lines[lineid]:
                    lineid += 1
                    continue

                while lineid < len(lines) and (' UTC [' in lines[lineid] or 'CST [' in lines[lineid]):
                    if 'duration' in lines[lineid]:
                        duration = pattern_num.findall(lines[lineid])[-1]
                    else:
                        lineid += 1
                        continue
                    plan_strs = []
                    lineid += 1
                    while lineid < len(lines) and (' UTC [' not in lines[lineid] and 'CST [' not in lines[lineid]):
                        plan_strs.append(lines[lineid])
                        lineid += 1
                    if plan_strs != []:
                        plan_obj = json.loads(s=''.join(plan_strs))
                        plan_obj['Plan']['Planning Time'] = float(duration) - plan_obj['Plan']['Actual Total Time']
                        plan_obj['Plan']['Execution Time'] = float(duration)
                        json_obj.append(plan_obj['Plan'])
        return json_obj

    def get_input(self, data):  # Helper for sample_data
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
        new_samp_dict = {}
        new_samp_dict["node_type"] = data["Node Type"]
        new_samp_dict["subbatch_size"] = len(data)

        feat_vec = np.array(
            [self.input_func[jss["Node Type"]](jss) for jss in data])

        # normalize feat_vec
        feat_vec = norm_model.fit_transform(feat_vec)

        total_time = [jss['Actual Total Time'] for jss in data]
        total_cost = [jss['Total Cost'] for jss in data]
        if 'Planning Time' in data:
            plan_time = [jss['Planning Time'] for jss in data]
            new_samp_dict["plan_time"] = np.array(plan_time).astype(np.float32)

        child_plan_lst = []
        if 'Plans' in data:
            for i in range(len(data['Plans'])):
                child_plan_dict = self.get_input([jss['Plans'][i] for jss in data])
                child_plan_lst.append(child_plan_dict)

        new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
        new_samp_dict["children_plan"] = child_plan_lst
        new_samp_dict["total_time"] = np.array(total_time).astype(np.float32)
        new_samp_dict["total_cost"] = np.array(total_cost).astype(np.float32)

        if 'Subplan Name' in data:
            new_samp_dict['is_subplan'] = True
        else:
            new_samp_dict['is_subplan'] = False
        return new_samp_dict
