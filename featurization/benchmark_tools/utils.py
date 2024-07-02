import json
import os
import pickle
import re
from types import SimpleNamespace


def load_schema_json(dataset):
    schema_path = os.path.join('featurization/datasets/', dataset, 'schema.json')
    assert os.path.exists(schema_path), f"Could not find schema.json ({schema_path})"
    return load_json(schema_path)


def load_column_statistics(dataset, namespace=True):
    path = os.path.join('featurization/datasets/', dataset.split("_")[0], 'column_statistics.json')
    assert os.path.exists(path), f"Could not find file ({path})"
    return load_json(path, namespace=namespace)


def load_string_statistics(dataset, namespace=True):
    path = os.path.join('featurization/datasets/', dataset.split("_")[0], 'string_statistics.json')
    assert os.path.exists(path), f"Could not find file ({path})"
    return load_json(path, namespace=namespace)


def load_schema_sql(dataset, sql_filename):
    sql_path = os.path.join('featurization/datasets/', dataset.split("_")[0], 'schema_sql', sql_filename)
    assert os.path.exists(sql_path), f"Could not find schema.sql ({sql_path})"
    with open(sql_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


def load_json(path, namespace=True):
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = json.load(json_file)
    return json_obj


def load_queries(path):
    queries = []
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                queries.append(f.read())
    return queries


def load_database_stats(dataset):
    path = os.path.join('featurization/datasets/', dataset.split("_")[0], 'database_stats.pickle')
    if not os.path.exists(path):
        assert False, f"Could not find database_stats.pickle ({path})"
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_attr_table(dataset):
    path = os.path.join('featurization/datasets/', dataset.split("_")[0], 'attr_table.pickle')
    if not os.path.exists(path):
        assert False, f"Could not find attr_table.pickle ({path})"
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def load2json(filepath):
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
                    json_obj.append(plan_obj['Plan']['Actual Total Time'])
    return json_obj
