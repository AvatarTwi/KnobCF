import time
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration
from copy import deepcopy
from config import config


class BEST_CONFIG:
    def __init__(self, state, func, initial_design, n_iters, logging_dir=None):
        assert state.target_metric == 'throughput'
        self.func = func  # 调用虚拟机pg进行实际查询
        self.initial_design = initial_design
        self.input_space = initial_design.cs
        self.n_iters = n_iters
        self.logging_dir = logging_dir
        self.state = state

    def get_hyperparam_bound(self, name):
        for hyperparameter in self.input_space.get_hyperparameters():
            if hyperparameter.name != name:
                continue
            else:
                if isinstance(hyperparameter, CS.UniformFloatHyperparameter):
                    # 获取范围
                    lower_bound = hyperparameter.lower
                    upper_bound = hyperparameter.upper
                    bound = [lower_bound, upper_bound]
                    return bound
                elif isinstance(hyperparameter, CS.UniformIntegerHyperparameter):
                    lower_bound = hyperparameter.lower
                    upper_bound = hyperparameter.upper
                    bound = [lower_bound, upper_bound]
                    return bound

    def get_new_hyperspace(self, currspace):
        # 创建一个新的超参空间，用于之后的转换
        new_input_space = CS.ConfigurationSpace()
        for hyperparameter in currspace.get_hyperparameters():
            if '_CHANGED' in hyperparameter.name:
                temp_name = hyperparameter.name.replace('_CHANGED', '')
                categorical_hyperparam = CS.CategoricalHyperparameter(name=temp_name, choices=['on', 'off'],
                                                                      default_value='on' if hyperparameter.default_value == 1 else 'off')
                new_input_space.add_hyperparameter(categorical_hyperparam)
            else:
                new_input_space.add_hyperparameter(hyperparameter)
        return new_input_space

    def RBS(self, lhdesign, last_best_perf, bound_search_budget):
        prev_perf = self.state.default_perf
        assert prev_perf >= 0

        results = []

        # Bootstrap with random samples
        if lhdesign.configs is not None:
            lhdesign.configs = None  # 置空configs,重新进行选择
        # seed = np.random.rand()
        # lhdesign.rng.seed(int(seed))
        init_configurations = lhdesign.select_configurations()
        # print(lhdesign.cs)
        # print(init_configurations)
        time.sleep(5)
        best_perf = 0
        best_perf_pos = -1
        dim_value = {}
        dim_bound_next = {}
        for idx, key in enumerate(init_configurations[0].keys()):
            dim_value[key] = []
            dim_bound_next[key] = {}
        for i, knob_data in enumerate(init_configurations):

            print(knob_data)
            temp1 = {}
            for idx, key in enumerate(knob_data.keys()):  # 将连续的值转换为离散的on或者off
                if '_CHANGED' in key:
                    if knob_data[key] >= 1.0:
                        temp1[key.replace('_CHANGED', '')] = 'on'
                    else:
                        temp1[key.replace('_CHANGED', '')] = 'off'
                else:
                    temp1[key] = knob_data[key]

            # 使用前面定义的new_input_space
            knob1 = Configuration(configuration_space=self.get_new_hyperspace(lhdesign.cs), values=temp1)
            # print(knob1)
            # time.sleep(100)
            print(f'Iter {i} -- RANDOM')
            ### reward, metric_data = env.simulate(knob_data)
            # metrics & perf
            perf, metric_data = self.func(knob1)
            perf = -perf  # maximize
            assert perf >= 0

            for idx, key in enumerate(knob_data.keys()):
                dim_value[key].append((knob_data[key], i))

            if perf > best_perf:
                best_perf = perf
                best_perf_pos = i
            # compute reward
            reward = self.get_reward(perf, prev_perf)
            # LOG
            print(f'Iter {i} -- PERF = {perf}')
            print(f'Iter {i} -- METRICS = {metric_data}')
            print(f'Iter {i} -- REWARD = {reward}')

            prev_metric_data = metric_data
            prev_knob_data = knob_data.get_array()  # scale to [0, 1]
            prev_reward = reward
            prev_perf = perf

        for idx, key in enumerate(dim_value.keys()):  # 对当前LHD取得的样本进行值排序
            sorted_dim = {key: sorted(values, key=lambda x: x[0]) for key, values in dim_value.items()}

        for idx, key in enumerate(sorted_dim.keys()):  # 获取下一次RBS的每一维的范围
            for i, _tuple in enumerate(sorted_dim[key]):
                if _tuple[1] == best_perf_pos:
                    if i == len(sorted_dim[key]) - 1:
                        dim_bound_next[key]['l_bound'] = sorted_dim[key][i - 1][0]
                        dim_bound_next[key]['r_bound'] = self.get_hyperparam_bound(key)[1]
                    elif i == 0:
                        dim_bound_next[key]['l_bound'] = self.get_hyperparam_bound(key)[0]
                        dim_bound_next[key]['r_bound'] = sorted_dim[key][i + 1][0]
                    else:
                        dim_bound_next[key]['l_bound'] = sorted_dim[key][i - 1][0]
                        dim_bound_next[key]['r_bound'] = sorted_dim[key][i + 1][0]

        new_lhdesign = deepcopy(lhdesign)  # copy然后生成一个新对象
        input_dims = []
        for hyperparam in new_lhdesign.cs.get_hyperparameters():
            _type = str(type(hyperparam))
            if 'UniformFloatHyperparameter' in _type:
                try:
                    dim = CSH.UniformFloatHyperparameter(
                        name=hyperparam.name,
                        lower=dim_bound_next[hyperparam.name]['l_bound'],
                        upper=dim_bound_next[hyperparam.name]['r_bound'],
                        default_value=dim_bound_next[hyperparam.name]['l_bound'])
                except:
                    dim = CSH.UniformFloatHyperparameter(
                        name=hyperparam.name,
                        lower=dim_bound_next[hyperparam.name]['l_bound'],
                        upper=dim_bound_next[hyperparam.name]['r_bound'],
                        default_value=dim_bound_next[hyperparam.name]['l_bound'] + 1e-5)  # 防止异常报错
                input_dims.append(dim)

            if 'UniformIntegerHyperparameter' in _type:
                try:
                    dim = CSH.UniformIntegerHyperparameter(
                        name=hyperparam.name,
                        lower=dim_bound_next[hyperparam.name]['l_bound'],
                        upper=dim_bound_next[hyperparam.name]['r_bound'],
                        default_value=dim_bound_next[hyperparam.name]['l_bound'])
                except:
                    dim = CSH.UniformIntegerHyperparameter(
                        name=hyperparam.name,
                        lower=dim_bound_next[hyperparam.name]['l_bound'],
                        upper=dim_bound_next[hyperparam.name]['r_bound'],
                        # default_value=dim_bound_next[hyperparam.name]['l_bound'] + 1e-5)
                        default_value=dim_bound_next[hyperparam.name]['l_bound'] + 1)
                input_dims.append(dim)
        input_space = CS.ConfigurationSpace(name="input", seed=config.seed)
        input_space.add_hyperparameters(input_dims)
        new_lhdesign.cs = input_space

        bound_search_budget -= 1
        if best_perf >= last_best_perf:
            if bound_search_budget == 0:
                return best_perf, bound_search_budget
            else:
                print('----------------------')
                print('budget:', bound_search_budget)
                print('----------------------')
                perf_next, budget_next = self.RBS(new_lhdesign, best_perf, bound_search_budget)
                return perf_next, budget_next
        else:
            return last_best_perf, bound_search_budget

    def get_reward(self, perf, prev_perf):
        """ Reward calculation same as CDBTune paper -- Section 4.2 """

        def calculate_reward(delta_default, delta_prev):
            if delta_default > 0:
                reward = ((1 + delta_default) ** 2 - 1) * np.abs(1 + delta_prev)
            else:
                reward = - ((1 - delta_default) ** 2 - 1) * np.abs(1 - delta_prev)

            # no improvement over last evaluation -- 0 reward
            if reward > 0 and delta_prev < 0:
                reward = 0

            return reward

        if perf == self.state.worse_perf:
            return 0

        # perf diff from default / prev evaluation
        delta_default = (perf - self.state.default_perf) / self.state.default_perf
        # TODO: check if this is correct
        # delta_prev = (perf - prev_perf) / prev_perf
        delta_prev = (perf - prev_perf) / (prev_perf + 1e-8)

        return calculate_reward(delta_default, delta_prev)
