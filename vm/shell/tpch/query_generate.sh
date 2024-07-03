#!/bin/bash

export DSS_CONFIG=/opt/module/QPPNetData/tpch-kit/dbgen
export DSS_QUERY=$DSS_CONFIG/queries
export DSS_PATH=/opt/module/QPPNetData/data

# shellcheck disable=SC2164
cd /opt/module/QPPNetData/tpch-kit/dbgen

for ((i=1;i<=22;i++)); do
  ./qgen ${i} -r 2  > /opt/module/knob_estm/tpch_queries/q${i}.sql
done

# shellcheck disable=SC2164
cd /opt/module/knob_estm/tpch_queries
for ((i=1;i<=22;i++)); do
  sed 's/([0-9])//' q${i}.sql > temp${i}.sql
  sed 's/^limit.*;$//' temp${i}.sql > q${i}.sql
done
sed 's/([0-9])//' q15.sql > temp15.sql
sed 's/^limit.*;$//' temp15.sql > q15.sql

rm -rf /opt/module/knob_estm/tpch_queries/temp*.sql

