select '"usertable":[','"'||percentile_disc(0.5) within group (order by ycsb_value) ||'",','"'||percentile_disc(0.5) within group (order by ycsb_key) ||'",'||'],' from usertable;
