# NOTE: Default values are gathered from PostgreSQL-9.6 on CloudLab c220g5

from spaces.common import finalize_conf, unfinalize_conf

KNOBS = [
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
