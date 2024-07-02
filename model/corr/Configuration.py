import json
from copy import copy
from pathlib import Path

import numpy as np

from spaces.knobs_sum import KNOBS_SUM


class Configuration:
    def __init__(self):
        definition_fp = Path('spaces/definitions') / "postgres-14.4.json"
        with open(definition_fp, 'r') as f:
            definition = json.load(f)
        self.definition = definition
        self.encoded_input = []
        self.init_default_conf(definition)

    def init_default_conf(self, definition):
        self.default_values = dict()
        self.default_min = dict()
        self.default_max = dict()
        self.default_part = []
        for i, thing in enumerate(definition):
            name = thing['name']
            type = thing['type']
            default = thing['default']

            if name in KNOBS_SUM:
                if 'min' in thing:
                    min = thing['min']
                    self.default_min[name] = min
                    max = thing['max']
                    self.default_max[name] = max

                if type == 'enum':
                    self.default_part.extend([0, 1] if default == 'on' else [1, 0])
                else:
                    self.default_values[name] = (default - min) / (max - min)
                    # self.default_values[name] = 0
                    self.encoded_input.append(self.default_values[name])

    def transfer_conf(self, input):
        part_id = -2
        encoded_id = -1
        part = copy(self.default_part)
        encoded_input = copy(self.encoded_input)
        for i, thing in enumerate(self.definition):
            name = thing['name']
            type = thing['type']

            if name in KNOBS_SUM:
                if type == 'enum':
                    part_id += 2
                else:
                    encoded_id += 1

                if name in input:
                    if 'min' in thing:
                        min = thing['min']
                        max = thing['max']
                    if type == 'enum':
                        part[part_id:part_id + 2] = [0, 1] if input[name] == 'on' else [1, 0]
                    else:
                        value = (input[name] - min) / (max - min)
                        encoded_input[encoded_id] = value
        part_input = np.array(part).reshape(1, -1)
        encoded_input = np.array(encoded_input).reshape(1, -1)
        conf = np.concatenate((part_input, encoded_input), axis=1)
        return conf