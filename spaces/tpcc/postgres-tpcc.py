# NOTE: Default values are gathered from PostgreSQL-9.6 on CloudLab c220g5

from spaces.common import finalize_conf, unfinalize_conf

KNOBS = [
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
