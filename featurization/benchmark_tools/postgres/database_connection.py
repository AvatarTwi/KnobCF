import os
import time
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


class PostgresDatabaseConnection():
    def __init__(self, database=None, user=None, password=None, host=None, port=None):
        self._cursor = None
        self._conn = None
        self._database = database
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self._result = None
        self._server = None

    def transform_dicts(self, column_stats_names, column_stats_rows):
        return [{k: v for k, v in zip(column_stats_names, row)} for row in column_stats_rows]

    def collect_table_attr(self):
        # column stats
        stats_query = """
            select
                a.column_name, a.table_name
            from
                information_schema.columns as a
            where
                table_schema = 'public'
            ORDER BY table_name;
        """
        column_stats_rows = self.get_result(stats_query, include_column_names=False)

        attr_table = {}
        for row in column_stats_rows:
            attr_table[row[0]] = row[1]

        return attr_table

    def collect_db_statistics(self):
        # column stats
        stats_query = """
            SELECT s.tablename, s.attname, s.null_frac, s.avg_width, s.n_distinct, s.correlation, c.data_type 
            FROM pg_stats s
            JOIN information_schema.columns c ON s.tablename=c.table_name AND s.attname=c.column_name
            WHERE s.schemaname='public';
        """
        column_stats_names, column_stats_rows = self.get_result(stats_query, include_column_names=True)
        column_stats = self.transform_dicts(column_stats_names, column_stats_rows)

        # table stats
        table_stats_names, table_stats_rows = self.get_result(
            f"SELECT relname, reltuples, relpages from pg_class WHERE relkind = 'r';",
            include_column_names=True)
        table_stats = self.transform_dicts(table_stats_names, table_stats_rows)
        return dict(column_stats=column_stats, table_stats=table_stats)

    def get_result(self, sql, include_column_names=False, db_created=True):
        connection, cursor = self.get_cursor(db_created=db_created)
        cursor.execute(sql)
        records = cursor.fetchall()
        self.close_conn(connection, cursor)

        if include_column_names:
            return [desc[0] for desc in cursor.description], records

        return records

    def get_cursor(self, db_created=True):
        connection = psycopg2.connect(
            database=self._database,
            user=self._user,
            password=self._password,
            host=self._host,
            port=self._port)
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()
        return connection, cursor

    def close_conn(self, connection, cursor):
        if connection:
            cursor.close()
            connection.close()
