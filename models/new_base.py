
from abc import ABC, abstractmethod

import torch
from torch import nn


class KGEModel(nn.Module, ABC):

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size):
        super(KGEModel, self).__init__()
        if data_type == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)
        self.rel = nn.Embedding(sizes[1]+sizes[0], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)

    @abstractmethod
    def get_queries(self, queries):
        pass

    @abstractmethod
    def new_get_queries(self, queries, neighbors):
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        pass

    def score(self, lhs, rhs, eval_mode):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def new_score(self, lhs, rhs, nhs, phs, eval_mode):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        nhs_e, nhs_biases = nhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        score_n = self.similarity_score(nhs_e, phs, eval_mode)
        #score_n_r = self.similarity_score((nhs_e, lhs_e[1]), rhs_e, eval_mode)
        # need parameter to balance teo item
        # 0.7/0.3
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                #return lhs_biases + rhs_biases.t() + score
                #return lhs_biases + rhs_biases.t() + score + score_n_r
                return lhs_biases + rhs_biases.t() + score + score_n
            else:
                #return lhs_biases + rhs_biases + 0.7 * score + 0.3 * score_n
                #return lhs_biases + rhs_biases + 0.85 * score + 0.15 * score_n
                #return lhs_biases + rhs_biases + score + score_n
                return lhs_biases + rhs_biases + score + score_n
        else:
            return score

    def get_factors(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries, neighbors, position, eval_mode=False):
        lhs_e, lhs_biases = self.new_get_queries(queries, neighbors, position)
        #
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)
        factors = self.get_factors(queries)
        return predictions, factors
    
    def new_forward(self, queries, neighbors, position, eval_mode=False):
        lhs_e, lhs_biases, nhs_e, phs_e = self.new_get_queries(queries, neighbors)
        #
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        predictions = self.new_score((lhs_e, lhs_biases), (rhs_e, rhs_biases), nhs_e, phs_e, eval_mode)
        factors = self.get_factors(queries)
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000):
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks

    def new_get_ranking(self, queries, filters, neighbors, position, batch_size=1000):
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.new_get_queries(these_queries, neighbors, position)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks
    
    def new_get_ranking_for_new_score(self, queries, filters, neighbors, position, batch_size=1000):
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.new_get_queries(these_queries, neighbors, position)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                #print('here')
                scores = self.new_score((q[0], q[1]), candidates, (q[2], q[3]), q[4], eval_mode=True)
                targets = self.new_score((q[0], q[1]), rhs, (q[2], q[3]), q[4], eval_mode=False)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks

    def compute_metrics(self, examples, filters, batch_size=500):
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at

    def new_compute_metrics(self, examples, filters, neighbors, position, batch_size=500):
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.new_get_ranking(q, filters[m], neighbors, position, batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at
