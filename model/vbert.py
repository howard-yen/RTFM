# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torchvision import transforms
from model.reader import Model as Base
from transformers import VisualBertModel
from rtfm import featurizer as X

class Model(Base):

    # overriden from paper_film.py
    @classmethod
    def create_env(cls, flags, featurizer=None):
        return super().create_env(flags, featurizer=featurizer or X.Concat([X.Visual(), X.Language(), X.Text(), X.ValidMoves()]))

    def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb=False, disable_wiki=False):
        super().__init__(observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb, disable_wiki=disable_wiki)

        self.resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        self.image_fc = nn.Sequential(
            nn.Linear(self.resnet.fc.out_features, 2048),
            nn.Tanh(),
        )
        self.bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre", cache_dir=".cache")
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.drep),
            nn.Tanh(),
        )
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_size = 32 * room_height

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

    def get_visual_embeddings(self, image):
        image_tensor = self.preprocess(image.view(-1, 3, self.image_size, self.image_size))
        output = self.resnet(image_tensor)
        output = self.image_fc(output)
        return output.view(-1, 1, 2048)

    def fuse(self, inputs, cell, inv, wiki, task):
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "token_type_ids": inputs["token_type_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        for k, v in model_inputs.items():
            model_inputs[k] = v.view(-1, v.shape[-1])

        visual_embeds = self.get_visual_embeddings(inputs["world_image"].to(torch.float32))
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        model_inputs.update({
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask
        })

        outputs = self.bert(**model_inputs)
        last_hidden_state = outputs.pooler_output

        output = self.fc(last_hidden_state).view(-1, self.drep)
        # print(f"output from vbert")
        return output  # (T*B, drep)
