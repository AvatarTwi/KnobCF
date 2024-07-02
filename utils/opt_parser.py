import argparse
import time

import config
from spaces.knobs_sum import KNOBS_COMBINE, KNOBS_SUM
from utils import constant


def getParser(workload='', mode='',num_clusters=16,optimizer_type='ddpg',query_encoder_model='KnobCF(few-shot)'):
    parser = argparse.ArgumentParser(description='Arg Parser')

    ini_conf = 1
    conf = constant.conf_filepath[ini_conf][workload]

    type = 'train'
    workload = workload
    if "run_db" in mode:
        if type != 'base':
            specific = workload + '4' + type + str(ini_conf) + "-" + str(time.time()).split(".")[1]
        else:
            specific = workload + '4' + type
    else:
        if type != 'base':
            specific = workload + '4' + type + str(ini_conf)
        else:
            specific = workload + '4' + type

    iter = constant.iters[type]
    dbname = constant.db_name[workload]

    querynum = constant.query_num[workload]
    queriesnum = constant.queries_num[workload]

    parser.add_argument('--num_clusters', default=num_clusters)

    parser.add_argument('--optimizer_type', default=optimizer_type)

    parser.add_argument('--query_encoder_model', default=query_encoder_model)

    # type
    # parser.add_argument('--type', default='on')
    parser.add_argument('--type', default='')

    parser.add_argument('--knob_dim', default=22)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--conf_filepath', default=conf)
    parser.add_argument('--db_name', default=dbname)
    parser.add_argument('--query_num', default=querynum)
    parser.add_argument('--queries_num', default=queriesnum)
    parser.add_argument('--workload', default=workload)
    parser.add_argument('--specific', default=specific)
    parser.add_argument('--iters', default=iter)
    parser.add_argument('--mode', default=mode)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--db_conf', default='conf/db_conf.yaml')
    parser.add_argument('--vm_conf', default='conf/vm_conf.yaml')
    parser.add_argument('--collect_script', default='collect.sh')
    parser.add_argument('--collect_partial_script', default='collect_partial.sh')
    parser.add_argument('--remote_scriptdir', default='/opt/module/knob_estm/shell/')
    parser.add_argument('--remote_datadir', default='/opt/module/pgsql/data/')
    parser.add_argument('--local_confdir', default='vm/conf/')
    parser.add_argument('--local_datadir', default='vm/data/')
    parser.add_argument('--local_scriptdir', default='vm/shell/')
    parser.add_argument('--leaf_size', default=10)
    parser.add_argument('--n_estimators_size', default=63)
    parser.add_argument('--model_save_path', default='model/save_model/')
    parser.add_argument('--recompute_features', default=False)
    parser.add_argument('--recompute_dataset', default=True)

    return parser


def save_opt(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
