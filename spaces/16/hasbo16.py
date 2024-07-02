# NOTE: Default values are gathered from PostgreSQL-9.6 on CloudLab c220g5

from spaces.common import finalize_conf, unfinalize_conf

KNOBS = [
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
