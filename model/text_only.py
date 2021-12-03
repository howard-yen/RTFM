# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from model.reader import Model as Base
from transformers import BertModel
from rtfm import featurizer as X

class Model(Base):

    # overriden from paper_film.py
    @classmethod
    def create_env(cls, flags, featurizer=None):
        return super().create_env(flags, featurizer=featurizer or X.Concat([X.LanguageAll(), X.Text(), X.ValidMoves()]))

    def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb=False, disable_wiki=False):
        super().__init__(observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb, disable_wiki=disable_wiki)

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.hidden_states_length = 768
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_states_length, self.drep),
            nn.Tanh(),
        )

    def encode_inv(self, inputs):
        return None

    def encode_cell(self, inputs):
        return None

    def encode_wiki(self, inputs):
        return None

    def encode_task(self, inputs):
        return None
    
    def compute_aux_loss(self, inputs, cell, inv, wiki, task):
        T, *_ = inputs["task"].size()
        return torch.Tensor([0] * T).to(inputs["input_ids"].device)

    def fuse(self, inputs, cell, inv, wiki, task):
        inputs = {
            "input_ids": inputs["input_ids"].view(-1, inputs["input_ids"].shape[-1]),
            "token_type_ids": inputs["token_type_ids"].view(-1, inputs["input_ids"].shape[-1]),
            "attention_mask": inputs["attention_mask"].view(-1, inputs["input_ids"].shape[-1]),
        }

        outputs = self.bert(**inputs)
        last_hidden_state = outputs.pooler_output

        return self.fc(last_hidden_state)  # (T*B, drep)

