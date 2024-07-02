from multi_mode import select_mode
from utils.opt_parser import getParser

# 1. choose mode type
# run db: run_db, run_db_default
# mode = 'run_db'
# mode = 'run_db_default'
# train query encoder: train_query_encoder, eval_query_encoder
# mode = 'train_query_encoder'
# mode = 'eval_query_encoder'
# train knob estimator: train_knob_estimator, eval_knob_estimator, trans_knob_estimator
# mode = 'train_knob_estimator'
mode = 'eval_knob_estimator'
# mode = 'trans_knob_estimator'

# 2. choose workload: tpcc, tpch, ycsb, ycsb_b, job
workload = 'ycsb'

# 3. num_clusters: 8, 10, 12, 14, 16
num_clusters = 16

# 4. optimizer type：ddpg or smac
optimizer_type = 'ddpg'

# 5.query_encoder_model: default, KnobCF(few-shot), KnobCF, query_former
query_encoder_model = 'KnobCF(few-shot)'

if __name__ == '__main__':
    # for i in range(2):
    #     args = getParser(ini_conf=ini_conf, workload=workload, type=type, mode=mode).parse_args()
    #     select_mode(args)
    args = getParser(workload=workload, mode=mode,num_clusters=num_clusters,
                     optimizer_type=optimizer_type,query_encoder_model=query_encoder_model).parse_args()
    select_mode(args)
