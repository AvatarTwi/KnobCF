import json
import os
import re

from config import config
from featurization.benchmark_tools.postgres.parse_plan import parse_plans

from featurization.benchmark_tools.utils import load_database_stats, load_attr_table


def load2json_plan(filepath):
    json_obj = []
    pattern_num = re.compile(r'\d+\.?\d*')

    with open(filepath, "r") as f:
        lines = [line for line in f.readlines()]
        lineid = 0
        while lineid < len(lines):
            if ' UTC [' not in lines[lineid] and 'CST [' not in lines[lineid]:
                lineid += 1
                continue

            while lineid < len(lines) and (' UTC [' in lines[lineid] or 'CST [' in lines[lineid]):
                if 'duration' in lines[lineid]:
                    duration = pattern_num.findall(lines[lineid])[-1]
                else:
                    lineid += 1
                    continue
                plan_strs = []
                lineid += 1
                while lineid < len(lines) and (' UTC [' not in lines[lineid] and 'CST [' not in lines[lineid]):
                    plan_strs.append(lines[lineid])
                    lineid += 1
                if plan_strs != []:
                    plan_obj = json.loads(s=''.join(plan_strs))
                    json_obj.append(plan_obj['Plan'])
    return json_obj
