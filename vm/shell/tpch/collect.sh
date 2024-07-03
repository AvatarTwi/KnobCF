#!/bin/bash

# shellcheck disable=SC2164
cd /opt/module/knob_estm/query/tpch/

for i in "$@"; do
    psql -h /tmp -U postgres -d postgres -f q${i}.sql > /dev/null 2>&1
done