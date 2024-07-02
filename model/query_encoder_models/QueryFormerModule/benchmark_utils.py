from dataset.job.job_utils import JOB_GET_INPUT
from dataset.tpcc.tpcc_utils import TPCC_GET_INPUT
from dataset.tpch.tpch_utils import TPCH_GET_INPUT
from dataset.ycsb.ycsb_utils import YCSB_GET_INPUT

BENCHMARK_UTILS= {
    'tpch':TPCH_GET_INPUT,
    'tpcc':TPCC_GET_INPUT,
    'job':JOB_GET_INPUT,
    'ycsb':YCSB_GET_INPUT,
    'ycsb_b':YCSB_GET_INPUT,
    'ycsb_c':YCSB_GET_INPUT
}