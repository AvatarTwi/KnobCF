TPCC_KNOBS = [
    "bgwriter_delay",
    "bgwriter_lru_maxpages",
    "checkpoint_completion_target",
    "checkpoint_timeout",
    "deadlock_timeout",
    "default_statistics_target",
    "effective_cache_size",
    "effective_io_concurrency",
    "shared_buffers"
]

TPCH_KNOBS = [
    "shared_buffers",
    "effective_cache_size",
    "work_mem",
    "default_statistics_target",
    "random_page_cost",
    "cpu_tuple_cost",
    "cpu_index_tuple_cost",
    "cpu_operator_cost",
    "maintenance_work_mem",
    "checkpoint_timeout",
    "wal_buffers",
    "autovacuum",
]

YCSBA_KNOBS = [
    "default_statistics_target",
    "effective_cache_size",
    "commit_delay",
    "shared_buffers",
    "temp_buffers",
    "wal_buffers",
    "work_mem"
]

YCSBB_KNOBS = [
    "backend_flush_after",
]

JOB_KNOBS = [
    "autovacuum",
    "backend_flush_after",
    "bgwriter_delay",
    "checkpoint_timeout",
    "default_statistics_target",
    "effective_cache_size",
    "maintenance_work_mem",
    "max_wal_size",
    "random_page_cost",
    "shared_buffers",
    "temp_buffers",
    "wal_buffers",
    "work_mem"
]

KNOBS = JOB_KNOBS
KNOBS.extend(YCSBA_KNOBS)
KNOBS.extend(YCSBB_KNOBS)
KNOBS.extend(TPCC_KNOBS)
KNOBS.extend(TPCH_KNOBS)
k = set(KNOBS)
sorted_list = sorted(list(k))
sorted_list = ['"' + i + '"' for i in sorted_list]

KNOBS_SUM = [
    "autovacuum",
    "backend_flush_after",
    "bgwriter_delay",
    "bgwriter_lru_maxpages",
    "checkpoint_completion_target",
    "checkpoint_timeout",
    "commit_delay",
    "cpu_index_tuple_cost",
    "cpu_operator_cost",
    "cpu_tuple_cost",
    "deadlock_timeout",
    "default_statistics_target",
    "effective_cache_size",
    "effective_io_concurrency",
    "maintenance_work_mem",
    "max_wal_size",
    "random_page_cost",
    "shared_buffers",
    "temp_buffers",
    "wal_buffers",
    "work_mem"
]

KNOBS_COMBINE = [
    "default_statistics_target",
    "effective_cache_size",
    "shared_buffers",
]
