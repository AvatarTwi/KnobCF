import pandas as pd
import json
import re
import math
import numpy as np
from collections import defaultdict
import psycopg2
import time
from sqlalchemy import create_engine

from manager.linux_manager import LinuxConnector

schema = {'imdbload': {'title': ['t.id', 't.kind_id', 't.production_year'],
                       'movie_companies': ['mc.id',
                                           'mc.company_id',
                                           'mc.movie_id',
                                           'mc.company_type_id'],
                       'cast_info': ['ci.id', 'ci.movie_id', 'ci.person_id', 'ci.role_id'],
                       'movie_info_idx': ['mi_idx.id', 'mi_idx.movie_id', 'mi_idx.info_type_id'],
                       'movie_info': ['mi.id', 'mi.movie_id', 'mi.info_type_id'],
                       'movie_keyword': ['mk.id', 'mk.movie_id', 'mk.keyword_id']},
          'postgres': {'customer':
                    ['c_custkey',
                     'c_name',
                     'c_address',
                     'c_nationkey',
                     'c_phone',
                     'c_acctbal',
                     'c_mktsegment',
                     'c_comment'],
                'lineitem':
                    ['l_orderkey',
                     'l_partkey',
                     'l_suppkey',
                     'l_linenumber',
                     'l_quantity',
                     'l_extendedprice',
                     'l_discount',
                     'l_tax',
                     'l_returnflag',
                     'l_linestatus',
                     'l_shipdate',
                     'l_commitdate',
                     'l_receiptdate',
                     'l_shipinstruct',
                     'l_shipmode',
                     'l_comment'],
                'nation':
                    ['n_nationkey',
                     'n_name',
                     'n_regionkey',
                     'n_comment'],
                'orders':
                    ['o_orderkey',
                     'o_custkey',
                     'o_orderstatus',
                     'o_totalprice',
                     'o_orderdate',
                     'o_orderpriority',
                     'o_clerk',
                     'o_shippriority',
                     'o_comment'],
                'part':
                    ['p_partkey',
                     'p_name',
                     'p_mfgr',
                     'p_brand',
                     'p_type',
                     'p_size',
                     'p_container',
                     'p_retailprice',
                     'p_comment'],
                'partsupp':
                    ['ps_partkey',
                     'ps_suppkey',
                     'ps_availqty',
                     'ps_supplycost',
                     'ps_comment'],
                'region':
                    ['r_regionkey',
                     'r_name',
                     'r_comment'],
                'supplier':
                    ['s_suppkey',
                     's_name',
                     's_address',
                     's_nationkey',
                     's_phone',
                     's_acctbal',
                     's_comment']
            },
          'benchmarksql':{
        'bmsql_config': [
            "cfg_name",
            "cfg_value"
        ],
        'bmsql_customer':
            ["c_w_id",
             "c_d_id",
             "c_id",
             "c_discount",
             "c_credit",
             "c_last",
             "c_first",
             "c_credit_lim",
             "c_balance",
             "c_ytd_payment",
             "c_payment_cnt",
             "c_delivery_cnt",
             "c_street_1",
             "c_street_2",
             "c_city",
             "c_state",
             "c_zip",
             "c_phone",
             "c_since",
             "c_middle",
             "c_data", ],
        'bmsql_district':
            ["d_w_id",
             "d_id",
             "d_ytd",
             "d_tax",
             "d_next_o_id",
             "d_name",
             "d_street_1",
             "d_street_2",
             "d_city",
             "d_state",
             "d_zip"],
        'bmsql_history':
            ["hist_id",
             "h_c_id",
             "h_c_d_id",
             "h_c_w_id",
             "h_d_id",
             "h_w_id",
             "h_date",
             "h_amount",
             "h_data"],
        'bmsql_item':
            ["i_id",
             "i_name",
             "i_price",
             "i_data",
             "i_im_id"],
        'bmsql_new_order':
            ["no_w_id",
             "no_d_id",
             "no_o_id"],
        'bmsql_oorder':
            ["o_w_id",
             "o_d_id",
             "o_id",
             "o_c_id",
             "o_carrier_id",
             "o_ol_cnt",
             "o_all_local",
             "o_entry_d"],
        'bmsql_order_line':
            ["ol_w_id",
             "ol_d_id",
             "ol_o_id",
             "ol_number",
             "ol_i_id",
             "ol_delivery_d",
             "ol_amount",
             "ol_supply_w_id",
             "ol_quantity",
             "ol_dist_info"],
        'bmsql_stock':
            ["s_w_id",
             "s_i_id",
             "s_quantity",
             "s_ytd",
             "s_order_cnt",
             "s_remote_cnt",
             "s_data",
             "s_dist_01",
             "s_dist_02",
             "s_dist_03",
             "s_dist_04",
             "s_dist_05",
             "s_dist_06",
             "s_dist_07",
             "s_dist_08",
             "s_dist_09",
             "s_dist_10"],
        "bmsql_warehouse": ["w_id", "w_tax", "w_ytd", "w_zip", "w_state", "w_city", "w_street_2", "w_street_1",
                            "w_name"],
    }}

t2alias = {
    'imdbload': {'title': 't', 'movie_companies': 'mc', 'cast_info': 'ci',
                 'movie_info_idx': 'mi_idx', 'movie_info': 'mi', 'movie_keyword': 'mk'},
}

alias2t = {}

for k, v in t2alias.items():
    alias2t1 = {}
    for k1, v1 in v.items():
        alias2t1[v1] = k1
    alias2t[k] = alias2t1

def to_vals(data_list):
    for dat in data_list:
        val = dat[0]
        if val is not None: break
    try:
        float(val)
        return np.array(data_list, dtype=float).squeeze()
    except:
        #         print(val)
        res = []
        for dat in data_list:
            try:
                mi = dat[0].timestamp()
            except:
                mi = 0
            res.append(mi)
        return np.array(res)


def collect_hist_file(cur, db_name):
    hist_file = pd.DataFrame(columns=['table', 'column', 'bins', 'table_column'])
    for table, columns in schema[db_name].items():
        for column in columns:
            # cmd = 'select {} from {} as {}'.format(column, table, t2alias[db_name][table])
            cmd = 'select {} from {}'.format(column, table)
            cur.execute(cmd)
            col = cur.fetchall()
            col_array = to_vals(col)
            hists = np.nanpercentile(col_array, range(0, 101, 2), axis=0)
            res_dict = {
                'table': table,
                'column': column,
                'table_column': '.'.join((table, column)),
                'bins': hists
            }
            hist_file = hist_file._append(res_dict, ignore_index=True)
    hist_file.to_csv('hist_file.csv', index=False)


def collect_sample(cur, db_name):
    ## sampling extension
    cmd = 'CREATE EXTENSION tsm_system_rows'
    cur.execute(cmd)
    tables = list(schema[db_name].keys())
    sample_data = {}
    for table in tables:
        cur.execute("Select * FROM {} LIMIT 0".format(table))
        colnames = [desc[0] for desc in cur.description]

        ts = pd.DataFrame(columns=colnames)

        for num in range(1000):
            cmd = 'SELECT * FROM {} TABLESAMPLE SYSTEM_ROWS(1)'.format(table)
            cur.execute(cmd)
            samples = cur.fetchall()
            for i, row in enumerate(samples):
                ts.loc[num] = row

        sample_data[table] = ts

    engine = create_engine('postgresql://postgres:postgres@192.168.75.130:5432/' + db_name + '_sample')

    for k, v in sample_data.items():
        v['sid'] = list(range(1000))
        cmd = 'alter table {} add column sid integer'.format(k)
        cur.execute(cmd)
        v.to_sql(k, engine, if_exists='append', index=False)

    query_file = pd.read_csv('data/imdb/workloads/synthetic.csv', sep='#', header=None)
    query_file.columns = ['table', 'join', 'predicate', 'card']

    table_samples = []
    for i, row in query_file.iterrows():
        table_sample = {}
        preds = row['predicate'].split(',')
        for i in range(0, len(preds), 3):
            left, op, right = preds[i:i + 3]
            alias, col = left.split('.')
            table = alias2t[alias]
            pred_string = ''.join((col, op, right))
            q = 'select sid from {} where {}'.format(table, pred_string)
            cur.execute(q)
            sps = np.zeros(1000).astype('uint8')
            sids = cur.fetchall()
            sids = np.array(sids).squeeze()
            if sids.size > 1:
                sps[sids] = 1
            if table in table_sample:
                table_sample[table] = table_sample[table] & sps
            else:
                table_sample[table] = sps
        table_samples.append(table_sample)


if __name__ == '__main__':
    db_name = 'benchmarksql'
    conm = psycopg2.connect(database=db_name, user="postgres", host="192.168.75.130", password="postgres",
                            port="5432")
    conm.set_session(autocommit=True)
    cur = conm.cursor()
    collect_hist_file(cur,db_name)
    # collect_sample(cur)
    conm.close()
