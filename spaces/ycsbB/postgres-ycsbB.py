# NOTE: Default values are gathered from PostgreSQL-9.6 on CloudLab c220g5

from spaces.common import finalize_conf, unfinalize_conf

# KNOBS = [
#     "backend_flush_after",
# ]
KNOBS = [
    "default_statistics_target",
    "effective_cache_size",
    "commit_delay",
    "shared_buffers",
    "temp_buffers",
    "wal_buffers",
    "work_mem"
]

