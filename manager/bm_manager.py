import pickle

import numpy as np
import pandas as pd

from config import config
from featurization.benchmark_tools.utils import load2json
from manager.linux_manager import LinuxConnector
from utils.timeout import timeout


class BenchmarkManager:
    def __init__(self, args):
        self.args = args
        self.lc = LinuxConnector(args.vm_conf)
        self.collect_script = args.collect_script
        self.collect_partial_script = args.collect_partial_script
        self.remote_scriptdir = args.remote_scriptdir
        self.remote_datadir = args.remote_datadir

        self.local_scriptdir = args.local_scriptdir + config['benchmark_info']['workload'] + '/'
        self.local_datadir = args.local_datadir + f'{self.args.optimizer_type}.{self.args.query_encoder_model}.{self.args.num_clusters}.{self.args.specific}.{self.args.type}' + '/'
        self.local_confdir = args.local_confdir + f'{self.args.optimizer_type}.{self.args.query_encoder_model}.{self.args.num_clusters}.{self.args.specific}.{self.args.type}' + '/'

        self.query_num = 0

    @timeout(12000)
    def collect_data(self, query_num):
        arr = [str(i) for i in range(1,query_num+1)]
        command = 'cd ' + self.remote_scriptdir + ' ; ./' + self.collect_script + ' ' + ' '.join(arr)
        res, err = self.lc.exec_command(command)
        return

    @timeout(12000)
    def collect_partial_data(self, arr):
        command = 'cd ' + self.remote_scriptdir + ' ; ./' + self.collect_partial_script + ' ' + ' '.join(arr)
        res, err = self.lc.exec_command(command)
        return

    def update_script(self, db_name):
        print("update script")
        self.lc.upload_file_path(self.remote_scriptdir, self.local_scriptdir + 'collect.sh')
        self.lc.upload_file_path(self.remote_scriptdir, self.local_scriptdir + 'collect_partial.sh')
        self.lc.exec_command('chmod 777 ' + self.remote_scriptdir + '*')
        self.lc.exec_command(f"psql -h /tmp -U postgres -d {db_name} -c \"analyse\"")
        print("update script finish")
        return

    def get_dbms_metrics(self):
        """ Parses DBMS metrics and returns their mean as a numpy array

        NOTE: Currently only DB-wide metrics are parsed; not table-wide ones
        """
        GLOBAL_STAT_VIEWS = ['pg_stat_bgwriter', 'pg_stat_database']
        lc = LinuxConnector(self.args.vm_conf)
        try:
            result, err = lc.exec_command(
                f"psql -h /tmp -U postgres -d postgres -c \"select * from pg_stat_bgwriter\"")
            row = result.split('\n')
            metric_name = [i.strip() for i in row[0].split('|')]
            metric_value1 = [float(i.strip()) for i in row[2].split('|') if i.strip().isnumeric()]
            result, err = lc.exec_command(
                f"psql -h /tmp -U postgres -d postgres -c \"select * from pg_stat_database where datname = 'postgres'\"")

            row = result.split('\n')
            metric_name = [i.strip() for i in row[0].split('|')][2:]
            metric_value2 = [float(i.strip()) for i in row[2].split('|') if i.strip().isnumeric()][1:]
        except Exception as err:
            return np.zeros(config.num_dbms_metrics)

        samples = {
            'pg_stat_bgwriter': metric_value1,
            'pg_stat_database': metric_value2
        }

        try:
            global_dfs = []
            for k in GLOBAL_STAT_VIEWS:
                s = samples[k]
                v = [[l for l in s if l != None]]
                cols = [f'{k}_{idx}' for idx in range(len(v[0]))]

                df = pd.DataFrame(data=v, columns=cols)
                df.dropna(axis=1, inplace=True)
                df = df.select_dtypes(['number'])
                global_dfs.append(df)

            df = pd.concat(global_dfs, axis=1)
            metrics = df.mean(axis=0).to_numpy()

        except Exception as err:
            return np.zeros(config.num_dbms_metrics)

        if len(metrics) != config.num_dbms_metrics:
            return np.zeros(config.num_dbms_metrics)

        return metrics

    def save_config(self, iter, config):
        with open(self.local_confdir + 'config' + str(iter) + '.pickle', 'wb') as f:
            pickle.dump(config, f)
        return

    def download_data(self, iter):
        self.lc.download_file_path(self.remote_datadir + 'serverlog',
                                   self.local_datadir + 'serverlog' + str(iter) + ".txt")

        return

    def download_data_single(self, iter):
        self.lc.download_file_path(self.remote_datadir + 'serverlog',
                                   self.args.local_datadir + config['benchmark_info']['workload'] + '4base/'
                                   + 'serverlog' + str(iter) + ".txt")
        return

    def parse_data(self, iter):
        if self.query_num == 0:
            data = load2json(self.local_datadir + "serverlog" + str(iter) + ".txt")
            self.query_num = len(data)
            return True, data
        else:
            data = load2json(self.local_datadir + "serverlog" + str(iter) + ".txt")
            if len(data) == self.query_num:
                return True, data
            else:
                return False, data

    def parse_data_partial(self, iter, length):
        if self.query_num == 0:
            data = load2json(self.local_datadir + "serverlog" + str(iter) + ".txt")
            self.query_num = len(data)
            return True, data
        else:
            data = load2json(self.local_datadir + "serverlog" + str(iter) + ".txt")
            if len(data) == length:
                return True, data
            else:
                return False, data

    def close(self):
        self.lc.close()
        return
