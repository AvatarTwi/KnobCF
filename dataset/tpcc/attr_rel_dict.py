# all operators used in tpc-h
all_dicts = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Index Scan',
             'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan',
             'Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize',
             'Merge Join', 'Subquery Scan', 'Gather', 'BitmapAnd', 'Memoize'
    , 'ModifyTable', 'LockRows', 'Result']

join_types = ['semi', 'inner', 'anti', 'full', 'right', 'left']

parent_rel_types = ['inner', 'outer', 'subquery']

sort_algos = ['quicksort', 'top-n heapsort']

aggreg_strats = ['plain', 'sorted', 'hashed']

rel_names = ['bmsql_config', 'bmsql_customer', 'bmsql_district', 'bmsql_history',
             'bmsql_item', 'bmsql_new_order', 'bmsql_oorder',
             'bmsql_order_line', 'bmsql_stock', 'bmsql_warehouse']

index_names = ['bmsql_config_pkey',
               'bmsql_customer_idx1', 'bmsql_customer_pkey', 'c_district_fkey',
               'bmsql_district_pkey', 'd_warehouse_fkey',
               'bmsql_history_pkey', 'h_customer_fkey', 'h_district_fkey',
               'bmsql_item_pkey',
               'bmsql_new_order_pkey', 'no_order_fkey',
               'bmsql_oorder_idx1', 'bmsql_oorder_pkey', 'o_customer_fkey',
               'bmsql_order_line_pkey', 'ol_order_fkey', 'ol_stock_fkey',
               'bmsql_stock_pkey', 's_item_fkey', 's_warehouse_fkey',
               'bmsql_warehouse_pkey']

rel_attr_list_dict = \
    {
        'bmsql_config': [
            "cfg_name",
            "cfg_value"
        ],
        'bmsql_customer':
            ["c_w_id",
             "c_d_id",
             "c_id",
             "c_discount",
             "c_credit",
             "c_last",
             "c_first",
             "c_credit_lim",
             "c_balance",
             "c_ytd_payment",
             "c_payment_cnt",
             "c_delivery_cnt",
             "c_street_1",
             "c_street_2",
             "c_city",
             "c_state",
             "c_zip",
             "c_phone",
             "c_since",
             "c_middle",
             "c_data", ],
        'bmsql_district':
            ["d_w_id",
             "d_id",
             "d_ytd",
             "d_tax",
             "d_next_o_id",
             "d_name",
             "d_street_1",
             "d_street_2",
             "d_city",
             "d_state",
             "d_zip"],
        'bmsql_history':
            ["hist_id",
             "h_c_id",
             "h_c_d_id",
             "h_c_w_id",
             "h_d_id",
             "h_w_id",
             "h_date",
             "h_amount",
             "h_data"],
        'bmsql_item':
            ["i_id",
             "i_name",
             "i_price",
             "i_data",
             "i_im_id"],
        'bmsql_new_order':
            ["no_w_id",
             "no_d_id",
             "no_o_id"],
        'bmsql_oorder':
            ["o_w_id",
             "o_d_id",
             "o_id",
             "o_c_id",
             "o_carrier_id",
             "o_ol_cnt",
             "o_all_local",
             "o_entry_d"],
        'bmsql_order_line':
            ["ol_w_id",
             "ol_d_id",
             "ol_o_id",
             "ol_number",
             "ol_i_id",
             "ol_delivery_d",
             "ol_amount",
             "ol_supply_w_id",
             "ol_quantity",
             "ol_dist_info"],
        'bmsql_stock':
            ["s_w_id",
             "s_i_id",
             "s_quantity",
             "s_ytd",
             "s_order_cnt",
             "s_remote_cnt",
             "s_data",
             "s_dist_01",
             "s_dist_02",
             "s_dist_03",
             "s_dist_04",
             "s_dist_05",
             "s_dist_06",
             "s_dist_07",
             "s_dist_08",
             "s_dist_09",
             "s_dist_10"],
        "bmsql_warehouse": ["w_id", "w_tax", "w_ytd", "w_zip", "w_state", "w_city", "w_street_2", "w_street_1",
                            "w_name"],
    }
