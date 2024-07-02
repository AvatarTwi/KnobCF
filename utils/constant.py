import json
from pathlib import Path

query_execution_time = 0

conf_filepath = {
    1: {
        'tpcc': 'conf/iniconf/tpcc.reward.part.ini',
        'tpch': 'conf/iniconf/tpch.reward.part.ini',
        'ycsb': 'conf/iniconf/ycsb.reward.part.ini',
        'ycsb_b': 'conf/iniconf/ycsb_b.reward.part.ini',
        'ycsb_c': 'conf/iniconf/ycsb_c.reward.part.ini',
        'job': 'conf/iniconf/job.reward.part.ini',
    },
    2: 'conf/iniconf/hasbo12',
    3: 'conf/iniconf/hasbo16',
}

iters = {
    'train': 300,
    'base': 1,
    'test': 300,
}

db_name = {
    'tpcc': 'benchmarksql',
    'tpch': 'postgres',
    'job': 'imdbload',
    'ycsb': 'ycsb',
    'ycsb_b': 'ycsb_b',
    'ycsb_c': 'ycsb_c'
}

query_num = {
    'tpcc': 256,
    'tpch': 22,
    'job': 70,
    'ycsb': 1001,
    'ycsb_b': 1001,
    'ycsb_c': 1000
}

queries_num = {
    'tpcc': 196,
    'tpch': 22,
    'job': 70,
    'ycsb': 974,
    'ycsb_b': 1000,
    'ycsb_c': 999,
    # 'ycsb': 1001,
    # 'ycsb_b': 1001
}



all_operators = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Index Scan',
                 'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan',
                 'Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize',
                 'Merge Join', 'Subquery Scan', 'Gather', 'BitmapAnd', 'Memoize'
    , 'ModifyTable', 'LockRows', 'Result', 'Append', 'Unique']


def knob_split_type():
    definition_fp = Path('../spaces/definitions') / "postgres-14.4.json"
    with open(definition_fp, 'r') as f:
        definition = json.load(f)
    num_name = [d['name'] for d in definition if d['type'] == 'real' or d['type'] == 'integer']
    str_name = [d['name'] for d in definition if d['type'] == 'enum']

    return num_name, str_name


train_name = [
    'autovacuum_analyze_scale_factor', 'autovacuum_analyze_threshold',
    'autovacuum_freeze_max_age',
    'autovacuum_max_workers', 'autovacuum_multixact_freeze_max_age', 'autovacuum_naptime',
    'autovacuum_vacuum_cost_delay', 'autovacuum_vacuum_cost_limit', 'autovacuum_vacuum_insert_scale_factor',
    'autovacuum_vacuum_insert_threshold', 'autovacuum_vacuum_scale_factor', 'autovacuum_vacuum_threshold',
    'autovacuum_work_mem', 'backend_flush_after', 'bgwriter_delay', 'bgwriter_flush_after',
    'bgwriter_lru_maxpages',
    'checkpoint_flush_after', 'checkpoint_timeout', 'commit_delay', 'commit_siblings', 'cpu_index_tuple_cost',
    'cpu_operator_cost', 'cpu_tuple_cost', 'cursor_tuple_fraction', 'deadlock_timeout',
    'default_statistics_target', 'effective_cache_size', 'effective_io_concurrency', 'from_collapse_limit',
    'geqo_effort', 'geqo_generations', 'geqo_pool_size', 'geqo_seed', 'geqo_selection_bias', 'geqo_threshold',
    'hash_mem_multiplier', 'jit_above_cost', 'jit_inline_above_cost', 'jit_optimize_above_cost',
    'join_collapse_limit', 'logical_decoding_work_mem', 'maintenance_io_concurrency', 'maintenance_work_mem',
    'max_connections', 'max_files_per_process', 'max_locks_per_transaction',
    'max_parallel_maintenance_workers',
    'max_parallel_workers', 'max_parallel_workers_per_gather', 'max_pred_locks_per_page',
    'max_pred_locks_per_relation', 'max_pred_locks_per_transaction', 'max_prepared_transactions',
    'max_stack_depth', 'max_wal_size', 'max_worker_processes', 'min_parallel_index_scan_size',
]

num_knobname = [
    # 'autovacuum_analyze_scale_factor','autovacuum_analyze_threshold',
    #  'autovacuum_freeze_max_age',
    # 'autovacuum_max_workers', 'autovacuum_multixact_freeze_max_age', 'autovacuum_naptime',
    # 'autovacuum_vacuum_cost_delay', 'autovacuum_vacuum_cost_limit', 'autovacuum_vacuum_insert_scale_factor',
    # 'autovacuum_vacuum_insert_threshold', 'autovacuum_vacuum_scale_factor', 'autovacuum_vacuum_threshold',
    # 'autovacuum_work_mem', 'backend_flush_after', 'bgwriter_delay', 'bgwriter_flush_after',
    # 'bgwriter_lru_maxpages',
    # 'bgwriter_lru_multiplier', '',
    # 'checkpoint_flush_after', 'checkpoint_timeout', 'commit_delay', 'commit_siblings', 'cpu_index_tuple_cost',
    # 'cpu_operator_cost', 'cpu_tuple_cost', 'cursor_tuple_fraction', 'deadlock_timeout',
    # 'default_statistics_target', 'effective_cache_size', 'effective_io_concurrency', 'from_collapse_limit',
    # 'geqo_effort', 'geqo_generations', 'geqo_pool_size', 'geqo_seed', 'geqo_selection_bias', 'geqo_threshold',
    # 'hash_mem_multiplier', 'jit_above_cost', 'jit_inline_above_cost', 'jit_optimize_above_cost',
    # 'join_collapse_limit', 'logical_decoding_work_mem', 'maintenance_io_concurrency', 'maintenance_work_mem',
    # 'max_connections', 'max_files_per_process', 'max_locks_per_transaction',
    # 'max_parallel_maintenance_workers',
    # 'max_parallel_workers', 'max_parallel_workers_per_gather', 'max_pred_locks_per_page',
    # 'max_pred_locks_per_relation', 'max_pred_locks_per_transaction', 'max_prepared_transactions',
    # 'max_stack_depth', 'max_wal_size', 'max_worker_processes',
    'min_parallel_table_scan_size', 'min_wal_size', 'old_snapshot_threshold', 'parallel_setup_cost',
    'parallel_tuple_cost', 'random_page_cost', 'seq_page_cost', 'shared_buffers', 'temp_buffers',
    # 'temp_file_limit', 'vacuum_cost_delay', 'vacuum_cost_limit', 'vacuum_cost_page_dirty',
    # 'vacuum_cost_page_hit', 'vacuum_cost_page_miss', 'vacuum_freeze_min_age', 'vacuum_freeze_table_age',
    # 'vacuum_multixact_freeze_min_age', 'vacuum_multixact_freeze_table_age', 'wal_buffers', 'wal_skip_threshold',
    'wal_writer_delay', 'wal_writer_flush_after', 'work_mem'
]

str_knobname = [
    'autovacuum', 'data_sync_retry', 'enable_bitmapscan', 'enable_gathermerge', 'enable_hashagg',
    'enable_hashjoin', 'enable_incremental_sort', 'enable_indexonlyscan', 'enable_indexscan', 'enable_material',
    'enable_mergejoin', 'enable_nestloop', 'enable_parallel_append', 'enable_parallel_hash',
    'enable_partition_pruning', 'enable_partitionwise_aggregate', 'enable_partitionwise_join', 'enable_seqscan',
    'enable_sort', 'enable_tidscan', 'full_page_writes', 'geqo', 'parallel_leader_participation',
    'quote_all_identifiers', 'wal_compression', 'wal_init_zero', 'wal_log_hints', 'wal_recycle'
]
