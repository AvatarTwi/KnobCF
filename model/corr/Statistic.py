import os.path

import matplotlib
import numpy as np
from scipy import stats

from featurization.benchmark_tools.utils import load2json
from utils.constant import queries_num

matplotlib.use('TkAgg')


class Statistic:

    @staticmethod
    def data_statistics_queries(workload):
        pass_key = []
        statis_std = {}
        statis_avg = {}
        statis_norm = {}
        statis_diff = {}
        statis_kurtosis = {}
        run_times = load2json('vm/data/' + workload + '4base' + '/serverlog0.txt')
        # run_times = load2json('../../vm/data/' + workload + '4base' + '/serverlog0.txt')

        time_dict = {i: [] for i in range(queries_num[workload])}
        for i, time in enumerate(run_times):
            time_dict[i % queries_num[workload]].append(time)
        for key, value in time_dict.items():
            if key not in statis_std.keys():
                statis_std[key] = []
                statis_avg[key] = []
                statis_kurtosis[key] = []
                statis_norm[key] = []
                statis_diff[key] = []

            value = np.array(value)
            q1 = np.percentile(value, 25)
            q3 = np.percentile(value, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            non_outlier_indices = (value >= lower_bound) & (value <= upper_bound)
            value = value[non_outlier_indices]
            statis_std[key].append(np.std(value))
            statis_avg[key].append(np.mean(value))

            statistic, p_value = stats.shapiro(value)
            statis_norm[key].append(p_value)

            if p_value > 0.001:
                # print(key, len([v for v in value if statis_avg[key][-1] - statis_std[key][-1] * 3 <
                #                 v < statis_avg[key][-1] + statis_std[key][-1] * 3]))
                statis_diff[key].append(statis_std[key][-1] * 3)
            else:
                # print(key, len([v for v in value if statis_avg[key][-1] - statis_std[key][-1] * 3 <
                #                 v < statis_avg[key][-1] + statis_std[key][-1] * 3]))
                # statis_diff[key].append(statis_std[key][-1] * 3)
                statis_diff[key].append(0)
                pass_key.append(key)

        return pass_key, statis_avg, statis_diff


if __name__ == '__main__':
    pass_key, statis_avg, statis_diff = Statistic.data_statistics_queries('ycsb_b')
    print(pass_key, statis_avg, statis_diff, len(statis_diff)-len(pass_key))

