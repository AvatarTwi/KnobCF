from config import config
from model.corr.knob_estimator import KnobEstimator

from run_optimizer import KnobOptimizer


def select_mode(args):
    query_encoder_model = args.query_encoder_model

    # init config
    config.update_from_file(args.workload, args.specific, args.iters, args.conf_filepath)
    config.seed = args.seed
    # number of DBMS internal metrics being sampled
    # config.num_dbms_metrics = 27
    config.num_dbms_metrics = 31

    # 和数据库互动，进行旋钮调优
    if args.mode == 'run_db' or 'run_db_default' in args.mode:
        KnobOptimizer(args)

    # 训练查询编码器 query encoder
    elif args.mode == 'train_query_encoder':
        model = KnobEstimator(args, query_encoder_model)
        model.c_pool.collect_data()
        model.train_query_encoder(500)

    elif args.mode == 'eval_query_encoder':
        model = KnobEstimator(args, query_encoder_model)
        model.query_encoder.load_model(args.workload)
        model.query_encoder.eval_query_encoder(args.workload)

    # 训练旋钮评估器 knob estimator
    elif args.mode == 'train_knob_estimator':
        model = KnobEstimator(args, query_encoder_model)
        model.init_knob_estimator(args.knob_dim + 256, args.num_clusters)
        if query_encoder_model == 'KnobCF(few-shot)':
            if args.type == 'on':
                benchmarks = []
                if config['benchmark_info']['workload'] == 'ycsb':
                    benchmarks.append('ycsb_b')
                if config['benchmark_info']['workload'] == 'ycsb_b':
                    benchmarks.append('ycsb')
                if config['benchmark_info']['workload'] == 'ycsb_c':
                    benchmarks.append('ycsb_b')
            else:
                benchmarks = []
                for b in ['tpch', 'tpcc', 'job', 'ycsb']:
                    if b == config['benchmark_info']['workload']:
                        continue
                    benchmarks.append(b)
        else:
            benchmarks = [config['benchmark_info']['workload']]
        model.query_encoder.load_model(args.workload)
        model.build_dataset(benchmarks)
        model.train_knob_estimator(500)

    elif args.mode == 'eval_knob_estimator':
        model = KnobEstimator(args, query_encoder_model)
        model.query_encoder.load_model(args.workload)
        model.build_dataset([config['benchmark_info']['workload']])
        model.load_model("train_knob_estimator_" + str(args.num_clusters) + "_" + str(args.query_encoder_model)
                         + "_" + config['benchmark_info']['workload'])
        # model.train_knob_estimator(200)
        # if query_encoder_model == 'KnobCF(few-shot)':
        #     benchmarks = []
        #     for b in ['tpch', 'tpcc', 'job', 'ycsb']:
        #         if b == config['benchmark_info']['workload']:
        #             continue
        #         benchmarks.append(b)
        # else:
        #     benchmarks = [config['benchmark_info']['workload']]
        # model.build_dataset(benchmarks)
        model.eval_knob_estimator(model.test_dataset)

    elif args.mode == 'trans_knob_estimator':
        model = KnobEstimator(args, query_encoder_model)
        model.c_pool.collect_data()
        model.build_dataset([config['benchmark_info']['workload']])
        model_name = 'tpch'
        model.load_model("knob_estimator_" + model_name)
        model.train_knob_estimator(50)
