import math
from decimal import Decimal

import numpy as np
import torch
from scipy.stats import stats


class MATH_UTIL:

    @staticmethod
    def MAX_VALUE():
        return math.inf

    @staticmethod
    def calculate_spearman_correlation_p(X, Y):
        return stats.spearmanr(X, Y)[1]

    @staticmethod
    def minkowski_distance(x, y, p_value):
        def p_root(value, root):
            my_root_value = 1 / float(root)
            return round(Decimal(value) ** Decimal(my_root_value), 3)
        return p_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)), p_value)

    @staticmethod
    def euclidean_distance(x, y):
        return MATH_UTIL.minkowski_distance(x, y, 2)

    @staticmethod
    def manhattan_distance(x, y):
        return MATH_UTIL.minkowski_distance(x, y, 1)

    @staticmethod
    def chebyshev_distance(x, y):
        return MATH_UTIL.minkowski_distance(x, y, float("inf"))

class Metric:
    @staticmethod
    def r_q(tt, pred_time, epsilon):
        """
            returns R(q) test loss defined in the QPP paper
        """
        rq_vec, _ = torch.max(
            torch.cat([((tt + epsilon) / (pred_time + epsilon)).unsqueeze(0),
                       ((pred_time + epsilon) / (tt + epsilon)).unsqueeze(0)], axis=0),
            axis=0)
        # print(rq_vec.shape)
        curr_rq = torch.mean(rq_vec).item()
        return curr_rq

    @staticmethod
    def r_q_numpy(tt, pred_time, epsilon):
        """
            returns R(q) test loss defined in the QPP paper
        """
        rq_vec = np.max(
            np.concatenate([np.expand_dims(((tt + epsilon) / (pred_time + epsilon)), axis=0),
                            np.expand_dims(((pred_time + epsilon) / (tt + epsilon)), axis=0)], axis=0),
            axis=0)
        # print(rq_vec.shape)
        curr_rq = np.mean(rq_vec)
        return curr_rq

    @staticmethod
    def mse(tt, pred_time, epsilon):
        return torch.pow((pred_time - tt), 2)

    @staticmethod
    def mse_numpy(tt, pred_time, epsilon):
        return np.power((pred_time - tt), 2)

    @staticmethod
    def pred_err(tt, pred_time, epsilon):
        """
            returns a vector of pred_err for each sample in the input
        """
        curr_pred_err = (torch.abs(tt - pred_time) + epsilon) / (tt + epsilon)
        return curr_pred_err

    @staticmethod
    def pred_err_numpy(tt, pred_time, epsilon):
        """
            returns a vector of pred_err for each sample in the input
        """
        curr_pred_err = (np.abs(tt - pred_time) + epsilon) / (tt + epsilon)
        return curr_pred_err

    @staticmethod
    def pred_err_sum(total_times, pred_times, epsilon):
        """
            returns a vector of pred_err for each sample in the input
        """
        pred_err = []
        for i, t in enumerate(total_times):
            curr_pred_err = (np.abs(t - pred_times[i]) + epsilon) / (t + epsilon)
            pred_err.append(curr_pred_err)
        return np.mean(pred_err)

    @staticmethod
    def accumulate_err(tt, pred_time, epsilon):
        """
            returns the pred_err for the sum of predictions
        """
        tt_sum = torch.sum(tt)
        pred_time_sum = torch.sum(pred_time)
        return torch.abs(pred_time_sum - tt_sum + epsilon) / (tt_sum + epsilon)

    @staticmethod
    def accumulate_err_numpy(tt, pred_time, epsilon):
        """
            returns the pred_err for the sum of predictions
        """
        tt_sum = np.sum(tt)
        pred_time_sum = np.sum(pred_time)
        return np.abs(pred_time_sum - tt_sum + epsilon) / (tt_sum + epsilon)

    @staticmethod
    def mean_mae(tt, pred_time, epsilon):
        """
            returns the absolute error for the mean of predictions
        """
        tt_mean = torch.mean(tt)
        pred_time_mean = torch.mean(pred_time)
        return torch.abs(pred_time_mean - tt_mean)

    @staticmethod
    def mean_mae_numpy(tt, pred_time, epsilon):
        """
            returns the absolute error for the mean of predictions
        """
        tt_mean = np.mean(tt)
        pred_time_mean = np.mean(pred_time)
        return np.abs(pred_time_mean - tt_mean)

    @staticmethod
    def q_error_numpy(true_time, pred_time, epsilon):
        """
            returns the absolute error for the mean of predictions
        """
        q_error = []

        a = []
        for idx, tt in enumerate(true_time):
            if pred_time[idx] < tt:
                a.append((tt + epsilon) / (pred_time[idx] + epsilon))
            else:
                a.append((pred_time[idx] + epsilon) / (tt + epsilon))

        for i in [99, 95, 90, 50]:
            q_error.append(np.percentile(a, i))
        q_error.append(np.mean(a))
        q_error.append(np.max(a))

        return q_error

    @staticmethod
    def q_error_all(true_time, pred_time, epsilon):
        """
            returns the absolute error for the mean of predictions
        """
        q_error = []
        for idx, tt in enumerate(true_time):
            if pred_time[idx] < tt:
                q_error.append((tt + epsilon) / (pred_time[idx] + epsilon))
            else:
                q_error.append((pred_time[idx] + epsilon) / (tt + epsilon))

        return torch.tensor(q_error)
