import pickle
import time

import numpy as np

from config import config
from manager.bm_manager import BenchmarkManager
from manager.db_manager import PgManager
from utils.constant import query_execution_time

times1 = True
times2 = True
times3 = True
time_num = 1

benchmark_sp = ['tpcc', 'ycsb', 'ycsb_b']


def prepare(args, sample):
    global time_num

    knob_dict = {key: value for key, value in sample.items() if not key.startswith('[')}

    pm = PgManager(args)
    pm.update_db_config(knob_dict)
    pm.close()

    time_num = 10 if config['benchmark_info']['workload'] in benchmark_sp else 1
    benchmark_manager = BenchmarkManager(args)
    benchmark_manager.update_script(args.db_name)
    if config['benchmark_info']['workload'] in benchmark_sp:
        for i in range(500):
            benchmark_manager.collect_data(args.query_num)
            # continue
    else:
        for i in range(50):
            benchmark_manager.collect_data(args.query_num)

    benchmark_manager.download_data_single(str(time.time()).split(".")[1])
    benchmark_manager.close()

    pm = PgManager(args)
    pm.restart_db()
    pm.close()


def get_reward(iter, args, sample):
    global time_num, query_execution_time

    result = {'throughput': 0, 'latency': 0, 'runtime': 0}

    knob_dict = {key: value for key, value in sample.items() if not key.startswith('[')}

    pm = PgManager(args)
    pm.update_db_config(knob_dict)
    pm.close()

    benchmark_manager = BenchmarkManager(args)

    temp_start_time = time.time()
    try:
        for i in range(time_num):
            benchmark_manager.collect_data(args.query_num)
    except TimeoutError as e:
        print(e)

    temp_end_time = time.time()
    query_execution_time += temp_end_time - temp_start_time
    print("query_execution_time:", query_execution_time)

    benchmark_manager.download_data(iter)
    metric = benchmark_manager.get_dbms_metrics()
    benchmark_manager.close()
    start_t = time.time()
    functional, query_runtimes = benchmark_manager.parse_data(iter)
    end_t = time.time()
    print("parse_data time:", {round((end_t - start_t)*1000)}, "ms")

    result['metrics'] = metric

    if functional:
        try:
            if config['benchmark_info']['workload'] in benchmark_sp:
                arr = np.array(query_runtimes).reshape(time_num, -1).T
                statis = []
                summery = 0.0
                for value in arr:
                    value = np.array(value)
                    q1 = np.percentile(value, 25)
                    q3 = np.percentile(value, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    non_outlier_indices = (value >= lower_bound) & (value <= upper_bound)
                    value = value[non_outlier_indices]
                    statis.append([np.mean(value), np.std(value)])
                    summery += np.mean(value)
                with open(
                        args.local_datadir + f'{args.optimizer_type}.{args.query_encoder_model}.{args.num_clusters}.{args.specific}.{args.type}' + '/'
                        + 'statis' + str(iter) + '.pickle', 'wb') as f:
                    pickle.dump(statis, f)
            else:
                summery = sum(query_runtimes)
            result['latency'] = summery  # ms
            result['throughput'] = float(benchmark_manager.query_num / (result['latency'] * time_num)) * 1000  # ops/s
        except Exception as e:
            print("Exception: ", e)
            result['latency'] = 100000000.0
            result['throughput'] = 0.0001  # ops/sec
    else:
        result['latency'] = 100000000.0  # ms
        result['throughput'] = 0.0001  # ops/sec
    return functional, result


def get_partial_reward(iter, args, sample, struct_known):
    global time_num

    result = {'throughput': 0, 'latency': 0, 'runtime': 0}

    knob_dict = {key: value for key, value in sample.items() if not key.startswith('[')}

    pm = PgManager(args)
    pm.update_db_config(knob_dict)
    pm.close()

    benchmark_manager = BenchmarkManager(args)

    start_t = time.time()
    try:
        for i in range(time_num):
            benchmark_manager.collect_partial_data(struct_known["query_need_collected"])
    except TimeoutError as e:
        print(e)
    end_t = time.time()
    print("parse_data time:", {round((end_t - start_t)*1000)}, "ms")

    benchmark_manager.download_data(iter)
    metric = benchmark_manager.get_dbms_metrics()
    benchmark_manager.close()
    functional, query_runtimes = benchmark_manager.parse_data_partial(iter, len(struct_known["query_need_collected"]))

    result['metrics'] = metric

    if functional:
        try:
            j = 0
            if config['benchmark_info']['workload'] in benchmark_sp:
                arr = np.array(query_runtimes).reshape(time_num, -1).T
                statis = []
                for value in arr:
                    value = np.array(value)
                    q1 = np.percentile(value, 25)
                    q3 = np.percentile(value, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    non_outlier_indices = (value >= lower_bound) & (value <= upper_bound)
                    value = value[non_outlier_indices]
                    statis.append(np.mean(value))
            else:
                statis = query_runtimes

            for i, v in enumerate(struct_known['runtimes']):
                if v == 0.0:
                    struct_known['runtimes'][i] = statis[j]
                    j += 1

            result['latency'] = sum(struct_known['runtimes'])  # ms
            result['throughput'] = float(benchmark_manager.query_num / (result['latency'] * time_num)) * 1000  # ops/s
        except Exception as e:
            print("Exception: ", e)
            result['latency'] = 100000000.0
            result['throughput'] = 0.0001  # ops/sec
    else:
        result['latency'] = 100000000.0  # ms
        result['throughput'] = 0.0001  # ops/sec

    return functional, result


def get_single_reward(args, sample):
    result = {'throughput': 0, 'latency': 0, 'runtime': 0}

    knob_dict = {key: value for key, value in sample.items() if not key.startswith('[')}

    pm = PgManager(args)
    pm.update_db_config(knob_dict)
    pm.close()

    benchmark_manager = BenchmarkManager(args)

    # for i in range(10):
    #     benchmark_manager.collect_data(args.query_num)
    # benchmark_manager.download_data_single(0)

    for i in range(50):
        benchmark_manager.collect_data(args.query_num)
    benchmark_manager.download_data_single(0)

    benchmark_manager.close()

    return result
