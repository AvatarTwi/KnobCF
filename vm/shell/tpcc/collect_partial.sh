#!/bin/bash

# shellcheck disable=SC2164
cd /opt/module/knob_estm/query/

for line_number in "$@"; do
    sed -n "${line_number}p" "tpcc.sql" | while IFS= read -r line; do
        psql -h /tmp -U postgres -d benchmarksql -c "$line" > /dev/null 2>&1
    done
done