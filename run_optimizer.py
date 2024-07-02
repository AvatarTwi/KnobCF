import logging
import pickle
import random
import time
from copy import copy
from pathlib import Path

import pandas as pd

from model.corr.Interface import Interface
from reward.get_reward import get_reward, get_partial_reward, get_single_reward, prepare
from utils.path_utils import PathUtils

pd.set_option('display.max_columns', None)
import numpy as np

from config import config
from executors.executor import ExecutorFactory
from optimizer import get_ddpg_optimizer, get_smac_optimizer, get_bestconf_optimizer
from spaces.space import ConfigSpaceGenerator
from storage import StorageFactory


def fix_global_random_state(seed=None):
    random.seed(seed)
    np.random.seed(seed)


class ExperimentState:
    def __init__(self, dbms_info, benchmark_info, results_path: Path, target_metric: str):
        self.iter = 0
        self.best_conf = None
        self.best_confs = []
        self.best_indexs = []
        self.best_perf = None
        self.worse_perf = None
        self.default_perf_stats = None

        assert target_metric in ['throughput', 'latency'], \
            f'Unsupported target metric: {target_metric}'
        self.minimize = target_metric != 'throughput'
        self._target_metric = target_metric

        self._dbms_info = dbms_info
        self._benchmark_info = benchmark_info

        assert results_path.exists()
        self._results_path = str(results_path)

    @property
    def benchmark_info(self):
        return self._benchmark_info

    @property
    def dbms_info(self):
        return self._dbms_info

    @property
    def results_path(self) -> str:
        return self._results_path

    @property
    def target_metric(self) -> str:
        return self._target_metric

    @property
    def default_perf(self) -> float:
        return self.default_perf_stats[self.target_metric]

    def is_better_perf(self, perf, other):
        return (perf > other) if not self.minimize else (perf < other)

    def __str__(self):
        fields = ['iter', 'best_conf', 'best_perf', 'worse_perf',
                  'default_perf_stats', 'target_metric']
        return '<ExperimentState>:\n' + \
               '\n'.join([f'{f}: \t{getattr(self, f)}' for f in fields])


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('storage')

get_xxx_optimizer = {
    'ddpg': get_ddpg_optimizer,
    'smac': get_smac_optimizer,
    'bestconfig': get_bestconf_optimizer
}


class KnobOptimizer:
    def __init__(self, args):
        global query_execution_time
        self.args = args
        self.query_execution_time = 0

        if self.args.query_encoder_model != 'default':
            self.knob_estm_interface = Interface(self.args)

        self.history_conf = []

        # Set global random state
        fix_global_random_state(seed=config.seed)

        # init input & output space
        self.spaces = ConfigSpaceGenerator.from_config(args, config)

        self.target_metric = self.spaces.target_metric
        # init storage class
        perf_label = 'Throughput' if self.target_metric == 'throughput' else 'Latency'
        self.columns = ['Iteration', perf_label, 'Optimum', 'Runtime']

        benchmark, workload, specific = (
            config['benchmark_info']['name'], config['benchmark_info']['workload'],
            config['benchmark_info']['specific'])

        PathUtils.path_build(args.local_confdir +
                             f'{self.args.optimizer_type}.{self.args.query_encoder_model}.{self.args.num_clusters}.{self.args.specific}.{self.args.type}')
        PathUtils.path_build(args.local_datadir +
                             f'{self.args.optimizer_type}.{self.args.query_encoder_model}.{self.args.num_clusters}.{self.args.specific}.{self.args.type}')

        inner_path = Path(
            f'{args.optimizer_type}.{args.query_encoder_model}.{self.args.num_clusters}.{specific}.{self.args.type}') / f'seed{self.args.seed}'
        results_path = Path(config['storage']['outdir']) / inner_path
        PathUtils.path_del(results_path)

        self.storage = StorageFactory.from_config(config, columns=self.columns, inner_path=inner_path)

        # store dbms & benchmark info in experiment state object
        benchmark_info_config = config.benchmark_info
        dbms_info_config = config.dbms_info
        self.exp_state = ExperimentState(
            dbms_info_config, benchmark_info_config, results_path, self.target_metric)

        # Create a new optimizer
        self.optimizer = get_xxx_optimizer[args.optimizer_type](config, self.spaces,
                                                                self.evaluate_dbms_conf, self.exp_state)

        # init executor
        if self.args.optimizer_type == 'smac':
            self.executor = ExecutorFactory.from_config(config, self.spaces, self.storage)
        else:
            self.executor = ExecutorFactory.from_config(config, self.spaces, self.storage, parse_metrics=True,
                                                        num_dbms_metrics=config.num_dbms_metrics)

        # evaluate on default config
        default_config = self.spaces.get_default_configuration()

        prepare(args, default_config)

        start_time = time.time()
        config.start_time = start_time

        if 'default' in self.args.mode:
            get_single_reward(args, default_config)
            return

        logger.info('Evaluating Default Configuration')
        logger.debug(default_config)

        if self.args.optimizer_type == 'smac':
            perf = self.evaluate_dbms_conf(default_config, state=self.exp_state)
        elif self.args.optimizer_type == 'ddpg':
            perf, default_metrics = self.evaluate_dbms_conf(default_config, state=self.exp_state)
        elif self.args.optimizer_type == 'bestconfig':
            perf, default_metrics = self.evaluate_dbms_conf(default_config, state=self.exp_state)
        else:
            raise ValueError(f'Unknown optimizer type: {self.args.optimizer_type}')

        perf = perf if self.exp_state.minimize else -perf
        assert perf >= 0, \
            f'Performance should not be negative: perf={perf}, metric={self.target_metric}'
        # assert len(default_metrics) == config.num_dbms_metrics, \
        #     ('DBMS metrics number does not match with expected: '
        #      f'[ret={len(default_metrics)}] [exp={config.num_dbms_metrics}]')

        # set starting point for worse performance
        self.exp_state.worse_perf = perf * 4 if self.exp_state.minimize else perf / 4

        # Start optimization loop
        if args.optimizer_type == 'smac':
            self.optimizer.optimize()  # SMAC
        elif args.optimizer_type == 'ddpg':
            self.optimizer.run()  # DDPG
        elif args.optimizer_type == 'bestconfig':
            best_performance, currbudget = self.optimizer.RBS(self.optimizer.initial_design, 0, 30)
            if currbudget != 0:
                best_performance1 = self.handle_budget(currbudget, 0)  # 重启后的最佳性能
        else:
            raise ValueError(f'Unknown optimizer type: {args.optimizer_type}')

        end_time = time.time()
        logger.info(f'Total Time used was: {round(end_time - config.start_time)}s, '
                    f'{round(end_time - config.start_time) / 60}min, '
                    f'{round(end_time - config.start_time) / 3600}h')

        # Print final stats
        logger.info(f'\nBest Configuration:\n{self.exp_state.best_conf}')

        for i, conf in enumerate(self.exp_state.best_confs):
            logger.info(f'Configuration {i}: {conf}')
            _, result = get_reward(self.exp_state.best_indexs[i], self.args, conf)
            # if self.target_metric == 'throughput':
            #     logger.info(f'Throughput: {self.exp_state.best_perf} ops/sec')
            # else:
            #     logger.info(f'95-th Latency: {self.exp_state.best_perf} milliseconds')
            if self.target_metric == 'throughput':
                logger.info(f'{self.exp_state.best_indexs[i]} Throughput: {result["throughput"]} ops/sec')
            else:
                logger.info(f'95-th Latency: {result["latency"]} milliseconds')
        logger.info(f'Saved @ {self.storage.outdir}')

    def handle_budget(self, currbudget, best_perf):
        best_performance, last_budget = self.optimizer.RBS(self.optimizer.initial_design, 0, currbudget)
        if last_budget != 0:
            if best_performance >= best_perf:
                best_p1 = self.handle_budget(last_budget, best_performance)
                return best_p1
            else:
                best_p2 = self.handle_budget(last_budget, best_perf)
                return best_p2
        else:
            if best_performance >= best_perf:
                return best_performance
            else:
                return best_perf

    def evaluate_dbms_conf(self, sample, state=None):
        logger.info(f'\n\n{25 * "="} Iteration {state.iter:2d} {25 * "="}\n\n')
        logger.info('Sample from optimizer: ')
        logger.info(sample)

        with open(
                self.args.local_confdir + f'{self.args.optimizer_type}.{self.args.query_encoder_model}.{self.args.num_clusters}.{self.args.specific}.{self.args.type}'
                + "/knob_data_vector" + str(state.iter) + '.pickle', 'wb') as f:
            pickle.dump(sample, f)

        with open(
                self.args.local_confdir + f'{self.args.optimizer_type}.{self.args.query_encoder_model}.{self.args.num_clusters}.{self.args.specific}.{self.args.type}'
                + "/knob_data" + str(state.iter) + '.pickle', 'wb') as f:
            pickle.dump(sample, f)

        if state.iter > 0:  # if not default conf
            sample = self.spaces.unproject_input_point(sample)

        conf = self.spaces.finalize_conf(sample)
        logger.info(f'Evaluating Configuration:\n{conf}')

        if self.args.query_encoder_model != 'default':
            start_t = time.time()
            struct_known = self.knob_estm_interface.judge_uncertainty(sample)
            end_t = time.time()
            print(f'judge_uncertainty time: {round((end_t - start_t)*1000)}')
            valid, result = get_partial_reward(state.iter, self.args, conf, struct_known)
            if valid:
                self.knob_estm_interface.process(struct_known["runtimes"])
        else:
            _, result = get_reward(state.iter, self.args, conf)

        ## Send configuration task to Nautilus
        dbms_info = dict(
            name=state.dbms_info['name'],
            config=conf,
            version=state.dbms_info['version'],
            result=result
        )

        if self.args.optimizer_type == 'smac':
            perf_stats = self.executor.evaluate_configuration(dbms_info, state.benchmark_info)
        elif self.args.optimizer_type == 'ddpg' or self.args.optimizer_type == 'bestconfig':
            perf_stats, metrics = self.executor.evaluate_configuration(dbms_info, state.benchmark_info)
        else:
            raise ValueError(f'Unknown optimizer type: {self.args.optimizer_type}')

        logger.info(f'Performance Statistics:\n{perf_stats}')

        if state.default_perf_stats is None:
            state.default_perf_stats = perf_stats

        print(perf_stats)

        target_metric = state.target_metric
        if perf_stats is None:
            # Error while evaluating conf -- set some reasonable value for target metric
            runtime, perf = 0, state.worse_perf
        else:
            if target_metric == 'latency' and \
                    perf_stats['throughput'] < state.default_perf_stats['throughput']:
                # Throughput less than default -- invalid round
                runtime, perf = perf_stats['runtime'], state.worse_perf
            else:
                runtime, perf = perf_stats['runtime'], perf_stats[target_metric]

        logger.info(f'Evaluation took {runtime} seconds')
        if target_metric == 'throughput':
            logger.info(f'Throughput: {perf} ops/sec')
        elif target_metric == 'latency':
            logger.info(f'95-th Latency: {perf} milliseconds')
        else:
            raise NotADirectoryError()

        if (perf_stats is not None) and ('sample' in perf_stats):
            sample = perf_stats['sample']
            logger.info(f'Point evaluated was: {sample}')

        end_time = time.time()
        logger.info(f'Time used was: {round(end_time - config.start_time)}s, '
                    f'{round(end_time - config.start_time) / 60}min, '
                    f'{round(end_time - config.start_time) / 3600}h')

        # Keep best-conf updated
        if (state.best_perf is None) or state.is_better_perf(perf, state.best_perf):
            state.best_conf, state.best_perf = copy(sample), perf
            state.best_confs.append(sample)
            state.best_indexs.append(state.iter)
        if (state.worse_perf is None) or not state.is_better_perf(perf, state.worse_perf):
            state.worse_perf = perf

        # Update optimizer results
        self.storage.store_result_summary(
            dict(zip(self.columns, [state.iter, perf, state.best_perf, runtime])))
        state.iter += 1

        # Register sample to the optimizer -- optimizer always minimizes
        perf = perf if state.minimize else -perf
        if self.args.optimizer_type == 'smac':
            return perf
        elif self.args.optimizer_type == 'ddpg':
            return perf, metrics
        elif self.args.optimizer_type == 'bestconfig':
            return perf, metrics
        else:
            raise ValueError(f'Unknown optimizer type: {self.args.optimizer_type}')
