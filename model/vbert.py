# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from model.reader import Model as Base
from transformers import VisualBertModel
from rtfm import featurizer as X

class Model(Base):

    # overriden from paper_film.py
    @classmethod
    def create_env(cls, flags, featurizer=None):
        return super().create_env(flags, featurizer=featurizer or X.Concat([X.Visual(), X.Language()]))

    def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drnn_small, drep, pretrained_emb=False, disable_wiki=False):
        super().__init__(observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drnn_small, drep, pretrained_emb, disable_wiki=disable_wiki)

        self.bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
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

    def get_visual_embeddings(self, image):
        return None

    def fuse(self, inputs, cell, inv, wiki, task):
        inputs = {
            "input_ids": inputs["input_ids"],
            "token_type_ids": inputs["token_type_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        visual_embeds = get_visual_embeddings(inputs["world_image"])
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update({
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask
        })

        outputs = model(**inputszer)
        last_hidden_state = outputs.last_hidden_state

        return self.fc(last_hidden_state)  # (T*B, drep)
