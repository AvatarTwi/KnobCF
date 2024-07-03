#!/bin/bash

# shellcheck disable=SC2164
cd /opt/module/knob_estm/query/

for line_number in "$@"; do
    sed -n "${line_number}p" "ycsb.sql" | while IFS= read -r line; do
        psql -h /tmp -U postgres -d ycsb -c "$line" > /dev/null 2>&1
    done
done