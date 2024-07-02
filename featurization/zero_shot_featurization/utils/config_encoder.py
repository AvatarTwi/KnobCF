import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from featurization.zero_shot_featurization.utils.embeddings import EmbeddingInitializer
from featurization.zero_shot_featurization.utils.fc_out_model import FcOutModel



class KnobEncoderBase():
    """
    Model to encode one type of nodes in the graph (with particular knobs)
    """

    def __init__(self, device='cuda:0'):
        definition_fp = Path('spaces/definitions') / "postgres-14.4.json"
        with open(definition_fp, 'r') as f:
            definition = json.load(f)
        self.device = device
        self.definition = definition
        self.knob_idxs = []
        self.encoded_input = []
        # initialize embeddings and input dimension
        self.default_values = dict()
        self.input_dim = 0
        self.input_knob_idx = 0
        for i, thing in enumerate(definition):
            name = thing['name']
            type = thing['type']
            default = thing['default']

            if 'min' in thing:
                min = thing['min']
                max = thing['max']

            if type == 'enum':
                # similarly, a single value is encoded here
                self.knob_idxs.append(np.arange(self.input_knob_idx, self.input_knob_idx + 1))
                self.input_knob_idx += 1

                if default == 'on':
                    default = torch.tensor([1,0])
                elif default == 'off':
                    default = torch.tensor([0,1])
                self.default_values[name] = default
            else:
                # a single value is encoded here
                self.knob_idxs.append(np.arange(self.input_knob_idx, self.input_knob_idx + 1))
                self.input_knob_idx += 1
                self.default_values[name] = [(default - min) / (max - min)]
                self.input_dim += 1
            self.encoded_input.append(self.default_values[name])


    def forward(self, input):
        encoded_input = []
        for i, thing in enumerate(self.definition):
            name = thing['name']

            if name in input:
                type = thing['type']
                if 'min' in thing:
                    min = thing['min']
                    max = thing['max']
                if type == 'enum':
                    if input[name] == 'on':
                        value = [0,1]
                    elif input[name] == 'off':
                        value = [1,0]
                else:
                    value = [(input[name] - min) / (max - min)]

                self.encoded_input[i] = value
                # encoded_input.extend(value)

        encoded_input = np.concatenate(self.encoded_input, axis=0)

        return np.array(encoded_input)


class KnobEncoder(FcOutModel):
    """
    Model to encode one type of nodes in the graph (with particular knobs)
    """

    def __init__(self, definition, device='cuda:0', max_emb_dim=32, drop_whole_embeddings=False,
                 one_hot_embeddings=True, **kwargs):

        self.device = device
        self.definition = definition
        self.knob_idxs = []
        self.encoded_input = []
        # initialize embeddings and input dimension
        self.default_values = dict()
        self.input_dim = 0
        self.input_knob_idx = 0
        embeddings = dict()
        for i, thing in enumerate(definition):
            name = thing['name']
            type = thing['type']
            default = thing['default']

            if 'min' in thing:
                min = thing['min']
                max = thing['max']

            if type == 'enum':
                # similarly, a single value is encoded here
                self.knob_idxs.append(np.arange(self.input_knob_idx, self.input_knob_idx + 1))
                self.input_knob_idx += 1

                embd = EmbeddingInitializer(2, max_emb_dim, kwargs['p_dropout'],
                                            drop_whole_embeddings=drop_whole_embeddings, one_hot=one_hot_embeddings)
                embeddings[name] = embd
                if default == 'on':
                    default = torch.tensor(1)
                elif default == 'off':
                    default = torch.tensor(0)
                knob_data = torch.reshape(default, (-1,))
                self.default_values[name] = embeddings[name](knob_data.long())[0]
                self.input_dim += embd.emb_dim
            else:
                # a single value is encoded here
                self.knob_idxs.append(np.arange(self.input_knob_idx, self.input_knob_idx + 1))
                self.input_knob_idx += 1
                self.default_values[name] = torch.tensor([(default - min) / (max - min)])
                self.input_dim += 1
            self.encoded_input.append(torch.unsqueeze(self.default_values[name], 0))

        # super().__init__(input_dim=self.input_dim, **kwargs)
        super().__init__(input_dim=140, **kwargs)

        self.embeddings = nn.ModuleDict(embeddings)

    def forward(self, input):
        encoded_input = []
        for i, thing in enumerate(self.definition):
            name = thing['name']

            if name in input:
                type = thing['type']
                if 'min' in thing:
                    min = thing['min']
                    max = thing['max']
                if type == 'enum':
                    if input[name] == 'on':
                        temp_value = torch.tensor(1)
                    elif input[name] == 'off':
                        temp_value = torch.tensor(0)
                    knob_data = torch.reshape(temp_value, (-1,))
                    embd_data = self.embeddings[name](knob_data.long())
                    value = embd_data
                else:
                    value = torch.tensor([(input[name] - min) / (max - min)])

                # self.encoded_input[i] = torch.unsqueeze(value, 0)
                encoded_input.append(torch.unsqueeze(value, 0))

        # input_enc = torch.cat(self.encoded_input, dim=1).flatten()
        input_enc = torch.cat(encoded_input, dim=1).flatten()

        # return torch.unsqueeze(self.fcout(input_enc), 0)
        return torch.unsqueeze(input_enc,0)


class ConfigAggregator(FcOutModel):
    """
    Abstract message aggregator class. Combines child messages (either using MSCN or GAT) and afterwards combines the
    hidden state with the aggregated messages using an MLP.
    """

    def __init__(self, output_dim=0, **fc_out_kwargs):
        super().__init__(output_dim=output_dim, **fc_out_kwargs)

    def forward(self, feat):
        out = self.fcout(feat)
        return out
