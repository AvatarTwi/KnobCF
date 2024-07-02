import os

from featurization.benchmark_tools.postgres.database_connection import PostgresDatabaseConnection
import pickle

from utils.path_utils import PathUtils


def test_collect_db_statistics(dataset):
    # database=None, user=None, password=None, host=None, port=None
    db_statistics = PostgresDatabaseConnection(
        database=dataset,
        user='postgres',
        password='postgres',
        host="192.168.75.130",
        port=5432
    ).collect_db_statistics()
    PathUtils.path_build(dataset)
    path = os.path.join(dataset, 'database_stats.pickle')
    with open(path, 'wb') as f:
        pickle.dump(db_statistics, f)


def test_db_statistics(dataset):
    path = os.path.join(dataset, 'database_stats.pickle')
    with open(path, 'rb') as f:
        db_statistics = pickle.load(f)
    print(db_statistics)


def test_collect_table_attr(dataset):
    attr_table = PostgresDatabaseConnection(
        database=dataset,
        user='postgres',
        password='postgres',
        host="192.168.75.130",
        port=5432
    ).collect_table_attr()
    PathUtils.path_build(dataset)
    path = os.path.join(dataset, 'attr_table.pickle')
    with open(path, 'wb') as f:
        pickle.dump(attr_table, f)


def test_table_attr(dataset):
    path = os.path.join(dataset, 'attr_table.pickle')
    with open(path, 'rb') as f:
        db_statistics = pickle.load(f)
    print(db_statistics)


if __name__ == '__main__':
    # test_collect_db_statistics('imdbload')
    # test_db_statistics('job')
    # test_collect_table_attr('imdbload')
    test_table_attr('tpcc')
