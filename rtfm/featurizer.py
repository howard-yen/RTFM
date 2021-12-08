# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import revtok
import random
from pprint import pprint
from rtfm.dynamics import monster as M, item as I, world_object as O, event as E
from transformers import BertTokenizer
import pickle


class Featurizer:

    def get_observation_space(self, task):
        raise NotImplementedError()

    def featurize(self, task):
        raise NotImplementedError()


class Concat(Featurizer, list):

    def get_observation_space(self, task):
        feat = {}
        for f in self:
            feat.update(f.get_observation_space(task))
        return feat

    def featurize(self, task):
        feat = {}
        for f in self:
            feat.update(f.featurize(task))
        return feat


class ValidMoves(Featurizer):

    def can_move_to(self, agent, pos, world):
        x, y = pos
        existing = world.get_objects_at_pos((x, y))
        can_inhabit = all([agent.can_inhabit_cell(o) for o in existing])
        return 0 <= x < world.width and 0 <= y <= world.height and can_inhabit

    def get_observation_space(self, task):
        return {'valid': (len(M.BaseMonster.valid_moves), )}

    def featurize(self, task):
        valid = set(M.BaseMonster.valid_moves)
        if task.agent is not None and task.agent.position is not None:
            x, y = task.agent.position
            if not self.can_move_to(task.agent, (x-1, y), task.world):
                valid.remove(E.Left)
            if not self.can_move_to(task.agent, (x+1, y), task.world):
                valid.remove(E.Right)
            if not self.can_move_to(task.agent, (x, y-1), task.world):
                valid.remove(E.Up)
            if not self.can_move_to(task.agent, (x, y+1), task.world):
                valid.remove(E.Down)
        return {'valid': torch.tensor([a in valid for a in M.BaseMonster.valid_moves], dtype=torch.float)}


class Position(Featurizer):

    def get_observation_space(self, task):
        return {'position': (2, )}

    def featurize(self, task):
        feat = [0, 0]
        valid = set(M.BaseMonster.valid_moves)
        if task.agent is not None and task.agent.position is not None:
            x, y = task.agent.position
            feat = [x, y]
        return {'position': torch.tensor(feat, dtype=torch.long)}


class RelativePosition(Featurizer):

    def get_observation_space(self, task):
        return {'rel_pos': (task.world.height, task.world.width, 2)}

    def featurize(self, task):
        x_offset = torch.Tensor(task.world.height, task.world.width).zero_()
        y_offset = torch.Tensor(task.world.height, task.world.width).zero_()
        if task.agent is not None and task.agent.position is not None:
            x, y = task.agent.position
            for i in range(task.world.width):
                x_offset[:, i] = i - x
            for i in range(task.world.height):
                y_offset[i, :] = i - y
        return {'rel_pos': torch.stack([x_offset/task.world.width, y_offset/task.world.height], dim=2)}


class WikiExtract(Featurizer):

    def get_observation_space(self, task):
        return {
            'wiki_extract': (task.max_wiki, ),
        }

    def featurize(self, task):
        return {'wiki_extract': task.get_wiki_extract()}


class Progress(Featurizer):

    def get_observation_space(self, task):
        return {'progress': (1, )}

    def featurize(self, task):
        return {'progress': torch.tensor([task.iter / task.max_iter], dtype=torch.float)}


class Terminal(Featurizer):

    def get_observation_space(self, task):
        return {}

    def clear(self):
        # for windows
        if os.name == 'nt':
            _ = os.system('cls')
            # for mac and linux(here, os.name is 'posix')
        else:
            _ = os.system('clear')

    def featurize(self, task):
        self.clear()
        print("\r")
        print(task.world.render(perspective=task.perspective))

        print('-' * 80)
        print('Wiki')
        print(task.get_wiki())
        print('Task:')
        print(task.get_task())
        print('Inventory:')
        print(task.get_inv())

        print('-' * 80)
        print('Last turn:')
        print('-' * 80)
        if task.history:
            for event in task.history[-1]:
                print(event)

        print('-' * 80)
        print('Monsters:')
        print('-' * 80)
        for m in task.world.monsters:
            print('{}: {}'.format(m.char, m))
            print(m.describe())
            print()

        print('-' * 80)
        print('Items:')
        print('-' * 80)
        for m in task.world.items:
            print('{}: {}'.format(m.char, m))
            print(m.describe())
            print()

        print()
        pprint(M.Player.keymap)
        return {}


class Symbol(Featurizer):

    class_list = [
        O.Empty,
        O.Unobservable,

        O.Wall,
        O.Door,

        M.HostileMonster,

        M.QueuedAgent,
    ]

    class_map = {c: i for i, c in enumerate(class_list)}

    def __init__(self):
        self.num_symbols = len(self.class_list)

    def get_observation_space(self, task):
        return {
            'symbol': (*task.world_shape, task.max_placement),
        }

    def featurize(self, task):
        mat = task.world.get_observation(perspective=task.perspective, max_placement=task.max_placement)
        smat = []
        for y in range(0, len(mat)):
            row = []
            for x in range(0, len(mat[0])):
                os = mat[y][x]
                classes = [self.class_map[o.__class__] for o in os]
                row.append(classes)
            smat.append(row)
        return {'symbol': torch.tensor(smat, dtype=torch.long)}

class Language(Featurizer):

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_observation_space(self, task):
        return {
            # "inv_input_ids": (task.max_inv, ),
            # "wiki_input_ids": (task.max_wiki, ),
            # "task_input_ids": (task.max_task, ),
            # "inv_token_type_ids": (task.max_inv, ),
            # "wiki_token_type_ids": (task.max_wiki, ),
            # "task_token_type_ids": (task.max_task, ),
            # "inv_attention_mask": (task.max_inv, ),
            # "wiki_attention_mask": (task.max_wiki, ),
            # "task_attention_mask": (task.max_task, ),
            "input_ids": (task.max_inv + task.max_wiki + task.max_task, ),
            "token_type_ids": (task.max_inv + task.max_wiki + task.max_task, ),
            "attention_mask": (task.max_inv + task.max_wiki + task.max_task, ),
        }

    def featurize(self, task):
        task_text = task.get_task()
        wiki = task.get_wiki()
        inv = task.get_inv()

        # task_tokens = self.tokenizer(task_text, padding="max_length", max_length=task.max_task)
        # wiki_tokens = self.tokenizer(wiki, padding="max_length", max_length=task.max_wiki)
        # inv_tokens = self.tokenizer(inv, padding="max_length", max_length=task.max_inv)

        t_tokens = self.tokenizer.tokenize(task_text)
        w_tokens = self.tokenizer.tokenize(wiki)
        i_tokens = self.tokenizer.tokenize(inv)

        all_tokens = self.tokenizer(t_tokens, w_tokens + ["[SEP]"] + i_tokens, padding="max_length", max_length=task.max_task + task.max_wiki + task.max_inv, is_split_into_words=True, return_tensors="pt")

        ret ={
            "input_ids": all_tokens["input_ids"],
            "token_type_ids": all_tokens["token_type_ids"],
            "attention_mask": all_tokens["attention_mask"]
        }

        # ret = {
        #     "inv_input_ids": torch.tensor(inv_tokens["input_ids"]),
        #     "wiki_input_ids": torch.tensor(wiki_tokens["input_ids"]),
        #     "task_input_ids": torch.tensor(task_tokens["input_ids"]),
        #     "inv_token_type_ids": torch.tensor(inv_tokens["token_type_ids"]),
        #     "wiki_token_type_ids": torch.tensor(wiki_tokens["token_type_ids"]),
        #     "task_token_type_ids": torch.tensor(task_tokens["token_type_ids"]),
        #     "inv_attention_mask": torch.tensor(inv_tokens["attention_mask"]),
        #     "wiki_attention_mask": torch.tensor(wiki_tokens["attention_mask"]),
        #     "task_attention_mask": torch.tensor(task_tokens["attention_mask"]),
        # }

        return ret

class LanguageAll(Featurizer):

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # there are maximum five entities, each has at most 10 tokens. There is one more sentence for the walls, which should be only 10 tokens.
        self.max_world = 10 * 5 + 10

    def get_observation_space(self, task):
        return {
            "input_ids": (task.max_inv + task.max_wiki + task.max_task + self.max_world, ),
            "token_type_ids": (task.max_inv + task.max_wiki + task.max_task + self.max_world, ),
            "attention_mask": (task.max_inv + task.max_wiki + task.max_task + self.max_world, ),
        }

    def featurize(self, task):
        task_text = task.get_task()
        wiki = task.get_wiki()
        inv = task.get_inv()
        world = self.describe_world(task.world)

        t_tokens = self.tokenizer.tokenize(task_text)
        w_tokens = self.tokenizer.tokenize(wiki)
        i_tokens = self.tokenizer.tokenize(inv)
        wo_tokens = self.tokenizer.tokenize(world)

        all_tokens = self.tokenizer(t_tokens, w_tokens + ["[SEP]"] + i_tokens + ["[SEP]"] + wo_tokens, padding="max_length", max_length=task.max_task + task.max_wiki + task.max_inv + self.max_world, is_split_into_words=True, return_tensors="pt")

        ret ={
            "input_ids": all_tokens["input_ids"],
            "token_type_ids": all_tokens["token_type_ids"],
            "attention_mask": all_tokens["attention_mask"]
        }

        return ret

    def describe_world(self, world):
        desc = []
        for i in range(world.width):
            for j in range(world.height):
                for ob in world.map[(i, j)]:
                    if ob.name != "wall":
                        desc.append(f"{ob.name} is at row {j} column {i}.")
        desc.append("There are walls along the border.")

        return " ".join(desc)


class Visual(Featurizer):
    def __init__(self):
        super().__init__()
        self.image_size = 32
        self.transparency_threshold = 0.1
        with open("img_tensors.p", "rb") as f:
            self.imgs = pickle.load(f)

    def get_observation_space(self, task):
        return {
            "world_image": (3, task.world.height * self.image_size, task.world.width * self.image_size, ),
        }

    def featurize(self, task):
        world_map = task.world.map
        world_image = torch.zeros((task.world.height * self.image_size, task.world.width * self.image_size, 3), dtype=torch.float32)

        for i in range(task.world.width):
            for j in range(task.world.height):
                world_image[i*self.image_size:(i+1)*self.image_size, j*self.image_size:(j+1)*self.image_size] = self.imgs["empty"][:,:,:3]
                objects = world_map[(i, j)]
                for ob in objects:
                    ob_names = ob.name.split(" ")
                    if len(ob_names) > 1:
                        ob_name = ob_names[1]
                    else:
                        ob_name = ob_names[0]

                    ob_img = self.imgs[ob_name]
                    mask = ob_img[:,:,2] > self.transparency_threshold

                    world_image[i*self.image_size:i*self.image_size+mask.shape[0], j*self.image_size:j*self.image_size+mask.shape[1]][mask] = ob_img[:,:,:3][mask]

        if False:
            plt.imshow(world_image)
            plt.show()
            plt.savefig("worldimg.png")
            plt.clf()

        ret = {
            "world_image": world_image.permute(2, 0, 1),
        }

        return ret


class Text(Featurizer):

    def __init__(self, max_cache=1e6):
        super().__init__()
        self._cache = {}
        self.max_cache = max_cache

    def get_observation_space(self, task):
        return {
            'name': (*task.world_shape, task.max_placement, task.max_name),
            'name_len': (*task.world_shape, task.max_placement),
            'inv': (task.max_inv, ),
            'inv_len': (1, ),
            'wiki': (task.max_wiki, ),
            'wiki_len': (1, ),
            'task': (task.max_task, ),
            'task_len': (1, ),
        }

    def featurize(self, task, eos='pad', pad='pad'):
        mat = task.world.get_observation(perspective=task.perspective, max_placement=task.max_placement)
        smat = []
        lmat = []
        for y in range(0, len(mat)):
            srow = []
            lrow = []

            for x in range(0, len(mat[0])):
                names = []
                lengths = []
                for o in mat[y][x]:
                    n, l = self.lookup_sentence(o.describe(), task.vocab, max_len=task.max_name, eos=eos, pad=pad)
                    names.append(n)
                    lengths.append(l)
                srow.append(names)
                lrow.append(lengths)
            smat.append(srow)
            lmat.append(lrow)
        wiki, wiki_length = self.lookup_sentence(task.get_tokenized_wiki() if hasattr(task, 'get_tokenized_wiki') else task.get_wiki(), task.vocab, max_len=task.max_wiki, eos=eos, pad=pad)
        ins, ins_length = self.lookup_sentence(task.get_tokenized_task() if hasattr(task, 'get_tokenized_task') else task.get_task(), task.vocab, max_len=task.max_task, eos=eos, pad=pad)
        inv, inv_length = self.lookup_sentence(task.get_inv(), task.vocab, max_len=task.max_inv, eos=eos, pad=pad)
        ret = {
            'name': smat,
            'name_len': lmat,
            'inv': inv,
            'inv_len': [inv_length],
            'wiki': wiki,
            'wiki_len': [wiki_length],
            'task': ins,
            'task_len': [ins_length],
        }
        ret = {k: torch.tensor(v, dtype=torch.long) for k, v in ret.items()}
        return ret

    def lookup_sentence(self, sent, vocab, max_len=10, eos='pad', pad='pad'):
        if isinstance(sent, list):
            words = sent[:max_len-1] + [eos]
            length = len(words)
            if len(words) < max_len:
                words += [pad] * (max_len - len(words))
            return vocab.word2index([w.strip() for w in words]), length
        else:
            sent = sent.lower()
            key = sent, max_len
            if key not in self._cache:
                words = revtok.tokenize(sent)[:max_len-1] + [eos]
                length = len(words)
                if len(words) < max_len:
                    words += [pad] * (max_len - len(words))
                self._cache[key] = vocab.word2index([w.strip() for w in words]), length
                while len(self._cache) > self.max_cache:
                    keys = list(self._cache.keys())
                    del self._cache[random.choice(keys)]
            return self._cache[key]
