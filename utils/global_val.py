import json
import re
from pathlib import Path

knob = ''
sub = {}
pattern_num = re.compile(r'\d+\.?\d+')

def set_knob(new):
    global knob
    knob = new

def get_sub_conf():
    global sub
    definition_fp = Path('spaces/definitions') / "postgres-14.4.json"
    with open(definition_fp, 'r') as f:
        definition = json.load(f)
    static = {d['name']: (d['min'], d['max'], d['unit'])
           for d in definition
           if 'choices' not in d.keys()}
    for k, v in static.items():
        if '*' in str(v[2]):
            sub[k] = (v[1] - v[0]) * 8
        else:
            sub[k] = v[1] - v[0]

get_sub_conf()