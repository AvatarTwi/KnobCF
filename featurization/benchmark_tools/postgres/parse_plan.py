import collections
import re

import numpy as np
from tqdm import tqdm

import config
from featurization.benchmark_tools.generate_workload import LogicalOperator
from featurization.benchmark_tools.postgres.plan_operator import PlanOperator
from featurization.benchmark_tools.postgres.utils import plan_statistics


def parse_plan(analyze_plan):
    cur_operator = PlanOperator(analyze_plan)
    child_plan_lst = []

    if 'Plans' in analyze_plan:
        for i in range(len(analyze_plan['Plans'])):
            child_plan_dict = parse_plan(analyze_plan['Plans'][i])
            child_plan_lst.append(child_plan_dict)
    cur_operator.children = child_plan_lst

    return cur_operator


def parse_plans(run_stats):
    # keep track of column statistics
    column_id_mapping = dict()
    table_id_mapping = dict()
    partial_column_name_mapping = collections.defaultdict(set)

    attr_table = run_stats["attr_table"]

    database_stats = run_stats["database_stats"]
    # enrich column stats with table sizes
    table_sizes = dict()
    for table_stat in database_stats["table_stats"]:
        table_sizes[table_stat["relname"]] = table_stat["reltuples"]

    for i, column_stat in enumerate(database_stats["column_stats"]):
        table = column_stat["tablename"]
        column = column_stat["attname"]
        column_stat["table_size"] = table_sizes[table]
        column_id_mapping[(table, column)] = i
        partial_column_name_mapping[column].add(table)

    # similar for table statistics
    for i, table_stat in enumerate(database_stats["table_stats"]):
        table = table_stat["relname"]
        table_id_mapping[table] = i

    # parse individual queries
    parsed_plans = []
    avg_runtimes = []
    no_tables = []
    no_filters = []
    op_perc = collections.defaultdict(int)
    for i, q in tqdm(enumerate(run_stats["query_list"])):

        # compute average execution and planning times
        avg_runtime = run_stats["total_time_secs"][i]
        # parse the plan as a tree
        analyze_plan = parse_plan(q)
        analyze_plan.parse_lines_recursively(attr_table)
        analyze_plan.merge_recursively(analyze_plan)

        tables, filter_columns, operators = plan_statistics(analyze_plan)

        alias_dict = {
            'ci': 'cast_info',
            't': 'title',
            'mi':'movie_info',
            'mk':'movie_keyword',
            'mc':'movie_companies',
            'mi_idx':'movie_info_idx'
        }

        analyze_plan.parse_columns_bottom_up(column_id_mapping, partial_column_name_mapping, table_id_mapping,
                                             alias_dict=alias_dict)

        analyze_plan.plan_runtime = avg_runtime

        def augment_no_workers(p, top_no_workers=0):
            no_workers = p.plan_parameters.get('workers_planned')
            if no_workers is None:
                no_workers = top_no_workers

            p.plan_parameters['workers_planned'] = top_no_workers

            for c in p.children:
                augment_no_workers(c, top_no_workers=no_workers)

        augment_no_workers(analyze_plan)

        # collect statistics
        avg_runtimes.append(avg_runtime)
        no_tables.append(len(tables))
        for _, op in filter_columns:
            op_perc[op] += 1
        # log number of filters without counting AND, OR
        no_filters.append(len([fc for fc in filter_columns if fc[0] is not None]))

        parsed_plans.append(analyze_plan)

    parsed_runs = dict(parsed_plans=parsed_plans, database_stats=database_stats)

    stats = dict(
        runtimes=str(avg_runtimes),
        no_tables=str(no_tables),
        no_filters=str(no_filters)
    )

    return parsed_runs, stats
