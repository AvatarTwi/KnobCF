# NOTE: Default values are gathered from PostgreSQL-9.6 on CloudLab c220g5

from spaces.common import finalize_conf, unfinalize_conf

KNOBS = [
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