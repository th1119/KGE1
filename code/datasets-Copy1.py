import pickle
from typing import Dict, Tuple, List
import os

import numpy as np
import torch
import itertools
from models import KBCModel

class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)

        print(self.data['train'].shape)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()
        
        self.table = {}
        self.nums = {}
        self.neighbors = {}
        self.neighbors_addition = {}
        self.start = 0
        self.end = 5
        
        self.get_table()
        self.get_neighbors()
        self.get_nums()
        self.fix_neighbors()
        #self.random_fix_neighbors()
        self.addition_neighbors()

    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1

        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        return w

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))
    
    def get_table(self, ):
        for split in ["train", "test", "valid"]:
            self.table[split] = torch.zeros([self.n_entities, self.n_entities], dtype=torch.int64)
            for i in self.data[split]:
                self.table[split][i[0]][i[2]] = i[1]

    def get_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            index_rel = []
            for i_, i in enumerate(self.table[split]):
                non_zero_index = np.nonzero(i)
                non_zero_rel = i[non_zero_index]
                index_rel.append(torch.cat([non_zero_index, non_zero_rel], dim=1))
            self.neighbors[split] = index_rel

    def get_nums(self, ):
        for split in ["train", "test", "valid"]:
            length = [len(self.neighbors[split][j]) for j in range(len(self.neighbors[split]))]
            self.nums[split] = max(length)

    def fix_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            for j in range(len(self.neighbors[split])):
                if len(self.neighbors[split][j]) < self.nums[split]:
                    additional_data = torch.tensor([[j, self.n_predicates] for _ in range(self.nums[split]-len(self.neighbors[split][j]))])
                    self.neighbors[split][j] = torch.cat([self.neighbors[split][j], additional_data], dim=0)
            self.neighbors[split] = torch.tensor([n.tolist() for n in self.neighbors[split]])
            
    def random_fix_neighbors(self, mode=1):
        n_l = [1428.,18998.,30874.,36207.,38471.]
        n_l = torch.tensor([_l / 40943. for _l in n_l])
        ran = torch.rand(len(self.neighbors["train"]),1)
        logits = torch.gt(ran, n_l)
        for split in ["train", "test", "valid"]:
            for j in range(len(self.neighbors[split])):
                if len(self.neighbors[split][j]) < self.nums[split]:
                    additional_data = torch.tensor([[j, self.n_predicates] for _ in range(self.nums[split]-len(self.neighbors[split][j]))])
                    self.neighbors[split][j] = torch.cat([self.neighbors[split][j], additional_data], dim=0)
                if mode == 1:
                    self.neighbors[split][j] = self.neighbors[split][j].tolist()
                    self.neighbors[split][j][self.start:self.end] = [self.neighbors[split][j][_] if logits[j][_] else [j, self.n_predicates] for _ in range(self.start,self.end)]
                    self.neighbors[split][j] = torch.tensor(self.neighbors[split][j])
            self.neighbors[split] = torch.tensor([n.tolist() for n in self.neighbors[split]])

    def addition_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            tmp_node = self.neighbors[split][:, self.start:self.end, 0]
            f = lambda x: torch.tensor([_ for _ in itertools.permutations(x)])
            tmp = [f(_) for _ in tmp_node]
            tmp_ = [torch.unsqueeze(torch.cat([i for i in _], dim=0), dim=0) for _ in tmp]
            self.neighbors_addition[split] = torch.cat(tmp_, dim=0)

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
    ):
        model.eval()
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        flag = False
        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            # ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            ranks = model.get_ranking(q, self.neighbors[split], self.neighbors_addition[split], self.to_skip[m], batch_size=500)

            if log_result:
                if not flag:
                    results = np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)
                    flag = True
                else:
                    results = np.concatenate((results, np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)), axis=0)

            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
