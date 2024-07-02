import math
import re

from featurization.benchmark_tools.generate_workload import Aggregator, ExtendedAggregator, LogicalOperator
from featurization.benchmark_tools.postgres.parse_filter import parse_filter, PredicateNode
from featurization.benchmark_tools.postgres.utils import child_prod

estimated_regex = re.compile(
    '\(cost=(?P<est_startup_cost>\d+.\d+)..(?P<est_cost>\d+.\d+) rows=(?P<est_card>\d+) width=(?P<est_width>\d+)\)')
actual_regex = re.compile('\(actual time=(?P<act_startup_cost>\d+.\d+)..(?P<act_time>\d+.\d+) rows=(?P<act_card>\d+)')
op_name_regex = re.compile('->  ([^"(]+)')
workers_planned_regex = re.compile('Workers Planned: (\d+)')
# filter_columns_regex = re.compile('("\S+"."\S+")')
filter_columns_regex = re.compile('([^\(\)\*\+\-\'\= ]+)')
literal_regex = re.compile('(\'[^\']+\'::[^\'\)]+)')


class PlanOperator(dict):

    def __init__(self, plain_content, children=None, plan_parameters=None, plan_runtime=0):
        super().__init__()
        self.__dict__ = self
        self.plain_content = plain_content

        self.plan_parameters = plan_parameters if plan_parameters is not None else dict()
        self.children = list(children) if children is not None else []
        self.plan_runtime = plan_runtime

    def parse_lines(self, col_table_mapping):
        plan = self.plain_content

        op_name = plan["Node Type"]
        alias_dict=dict()

        # operator table
        if "Relation Name" in plan:
            table_name = plan["Relation Name"]

            if table_name.endswith('_pkey'):
                table_name = table_name.replace('_pkey', '')

            if 'Subquery Scan' in plan:
                alias_dict[table_name] = None
            else:
                self.plan_parameters.update(dict(table=table_name))

        self.plan_parameters.update(dict(op_name=op_name))

        self.plan_parameters.update({
            'est_startup_cost':float(plan["Startup Cost"]),
            'est_cost':float(plan["Total Cost"]),
            'est_card':float(plan["Plan Rows"]),
            'est_width':float(plan["Plan Width"]),
            'act_startup_cost':float(plan["Actual Startup Time"]),
            'act_time':float(plan["Actual Total Time"]),
            'act_card':float(plan["Actual Rows"]),
        })

        if 'Workers' in plan:
            self.plan_parameters.update(dict(workers_planned=len(plan["Workers"])))

        if 'Output' in plan:
            self.plan_parameters.update(dict(output_columns=self.parse_output_columns(plan["Output"],col_table_mapping)))

        if 'Filter' in plan:
            try:
                parse_tree = parse_filter(plan['Filter'], parse_baseline=False)
                self.add_filter(parse_tree)
            except:
                pass

        if 'Join Filter' in plan:
            parse_tree = parse_filter(plan['Join Filter'], parse_baseline=False)
            self.add_filter(parse_tree)

        if 'Hash Cond' in plan:
            parse_tree = parse_filter(plan['Hash Cond'], parse_baseline=False)
            self.add_filter(parse_tree)

        if 'Index Cond' in plan:
            parse_tree = parse_filter(plan['Index Cond'], parse_baseline=False)
            self.add_filter(parse_tree)

        if 'Inner Unique' in plan:
            self.plan_parameters.update(dict(inner_unique=True))


    def parse_output_columns(self, l, col_table_mapping):
        output_columns = []
        for col in l:
            # argument in a function call not an actual column
            if col.strip(')').isnumeric() or col in {'NULL::numeric', 'NULL::bigint', 'NULL::integer',
                                                     'NULL::double precision', 'NULL::text', 'NULL::bpchar'}:
                continue

            columns = []
            if 'count(*)' in col:
                agg = Aggregator.COUNT
            # for operators to change dates
            elif 'date_part(' in col:
                continue
            else:
                # remove literals
                col = re.sub(literal_regex, '', col)

                # timestamp types can be disregarded
                ts_text = '::timestamp without time zone'
                if ts_text in col:
                    col = col.replace(ts_text, '')

                for filter_m in filter_columns_regex.finditer(col):
                    curr_col = filter_m.group()
                    endpos = filter_m.span()[-1]

                    # check whether it is just an aggregation
                    if curr_col.lower() in ['avg', 'sum', 'count', 'min', 'max']:
                        # next character should be opening bracket
                        if len(col) > endpos and col[endpos] == '(':
                            continue

                    # additional PG keywords or operators
                    if curr_col.lower() in {'partial', 'precision', 'case', 'when', 'then', 'if', 'else', 'end',
                                            'or', '<>', '/', '~~', 'and', '"substring"', 'distinct'}:
                        continue
                    # just a type
                    if curr_col.startswith('::'):
                        continue
                    # just a literal
                    if curr_col.startswith("'") and curr_col.split("'")[-1].startswith("::") \
                            or curr_col.replace('.', '').isnumeric():
                        continue
                    try:
                        if '.' in curr_col:
                            columns.append(tuple(curr_col.split('.')))
                        else:
                            columns.append((col_table_mapping[curr_col], curr_col))
                    except:
                        pass
                # if there is an aggregation, find it
                agg = None
                if 'PARTIAL' in col:
                    col = col.replace('PARTIAL ', '').strip('( ')
                col = col.strip('(')
                for curr_agg in list(Aggregator) + list(ExtendedAggregator):
                    if col.startswith(str(curr_agg).lower()):
                        agg = str(curr_agg)

                # assert agg is not None, f"Could not parse {col}"
            output_columns.append(dict(aggregation=str(agg), columns=columns))
        return output_columns

    def add_filter(self, parse_tree):
        if parse_tree is not None:
            existing_filter = self.plan_parameters.get('filter_columns')
            if existing_filter is not None:
                parse_tree = PredicateNode(None, [existing_filter, parse_tree])
                parse_tree.operator = LogicalOperator.AND

            self.plan_parameters.update(dict(filter_columns=parse_tree))

    def parse_columns_bottom_up(self, column_id_mapping, partial_column_name_mapping, table_id_mapping,
                                alias_dict):
        if alias_dict is None:
            alias_dict = dict()

        # first keep track which tables are actually considered here
        node_tables = set()
        if self.plan_parameters.get('table') is not None:
            node_tables.add(self.plan_parameters.get('table'))

        for c in self.children:
            node_tables.update(
                c.parse_columns_bottom_up(column_id_mapping, partial_column_name_mapping, table_id_mapping, alias_dict))

        self.plan_parameters['act_children_card'] = child_prod(self, 'act_card')
        self.plan_parameters['est_children_card'] = child_prod(self, 'est_card')

        output_columns = self.plan_parameters.get('output_columns')
        if output_columns is not None:
            for output_column in output_columns:
                col_ids = []
                for c in output_column['columns']:
                    try:
                        c_id = self.lookup_column_id(c, column_id_mapping, node_tables, partial_column_name_mapping,
                                                     alias_dict)
                        col_ids.append(c_id)
                    except:
                        # not c[1].startswith('agg_')
                        if c[0] != 'subgb':
                            # raise ValueError(f"Did not find unique table for column {c}")
                            pass
                output_column['columns'] = col_ids

        filter_columns = self.plan_parameters.get('filter_columns')
        if filter_columns is not None:
            try:
                filter_columns.lookup_columns(self, column_id_mapping=column_id_mapping, node_tables=node_tables,
                                              partial_column_name_mapping=partial_column_name_mapping,
                                              alias_dict=alias_dict)
                self.plan_parameters['filter_columns'] = filter_columns.to_dict()
            except:
                pass

        # replace table by id
        table = self.plan_parameters.get('table')
        if table is not None:
            if table in table_id_mapping:
                self.plan_parameters['table'] = table_id_mapping[table]
            else:
                del self.plan_parameters['table']

        return node_tables

    def lookup_column_id(self, c, column_id_mapping, node_tables, partial_column_name_mapping, alias_dict):
        assert isinstance(c, tuple)
        # here it is clear which column is meant
        if len(c) == 2:
            table = c[0].strip('"')
            column = c[1].strip('"')

            if table in alias_dict:
                table = alias_dict[table]

                # this is a subquery and we cannot uniquely identify the corresponding table
                if table is None:
                    return self.lookup_column_id((c[1],), column_id_mapping, node_tables, partial_column_name_mapping,
                                                 alias_dict)

        # we now have to guess which table this column belongs to
        elif len(c) == 1:
            column = c[0].strip('"')

            potential_tables = partial_column_name_mapping[column].intersection(node_tables)
            assert len(potential_tables) == 1, f"Did not find unique table for column {column} " \
                                               f"(node_tables: {node_tables})"
            table = list(potential_tables)[0]
        else:
            raise NotImplementedError

        col_id = column_id_mapping[(table, column)]
        return col_id

    def merge_recursively(self, node):

        self.plan_parameters.update(node.plan_parameters)
        for self_c, c in zip(self.children, node.children):
            self_c.merge_recursively(c)

    def parse_lines_recursively(self,col_table_mapping):
        self.parse_lines(col_table_mapping)
        for c in self.children:
            c.parse_lines_recursively(col_table_mapping)

    def min_card(self):
        act_card = self.plan_parameters.get('act_card')
        if act_card is None:
            act_card = math.inf

        for c in self.children:
            child_min_card = c.min_card()
            if child_min_card < act_card:
                act_card = child_min_card

        return act_card

    def recursive_str(self, pre):
        pre_whitespaces = ''.join(['\t' for _ in range(pre)])
        # current_string = '\n'.join([pre_whitespaces + content for content in self.plain_content])
        current_string = pre_whitespaces + str(self.plan_parameters)
        node_strings = [current_string]

        for c in self.children:
            node_strings += c.recursive_str(pre + 1)

        return node_strings

    def __str__(self):
        rec_str = self.recursive_str(0)
        return '\n'.join(rec_str)