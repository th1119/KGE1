import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random as rd
import math
import time
import itertools
from functools import reduce

from models.new_base import KGEModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, logmap0, project, hyp_distance_multi_c

HYP_MODELS = ["GIE"]


class BaseH(KGEModel):

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1] + 1, 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        self.start = 0
        self.end = 5
        self.neighbor_length = self.end - self.start
        self.input_size = self.neighbor_length * reduce(lambda x, y: x * y, range(self.start + 1, self.end + 1))
        self.fc1 = nn.Linear(self.input_size, self.rank)
        self.fc2 = nn.Linear(self.rank, self.rank)
        self.fc3 = nn.Linear(self.rank, self.rank)
        self.fc4 = nn.Linear(self.rank, self.rank)
        self.ffc2 = nn.Linear(self.rank, self.input_size)
        self.ffc3 = nn.Linear(self.input_size, self.rank)
        self.norm = nn.LayerNorm(self.rank)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.fc_mn = nn.Linear(self.sizes[1], 1)
        if self.multi_c:
            c_init = torch.ones((self.sizes[1] + self.sizes[0], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1] + self.sizes[0], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1] + self.sizes[0], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1 = nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2


class GIE(BaseH):

    def __init__(self, args):
        super(GIE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.neighbor_vec = nn.Embedding(self.sizes[1] + 1, self.rank)
        self.neighbor_vec.weight.data = self.init_size * torch.randn((self.sizes[1] + 1, self.rank),
                                                                      dtype=self.data_type)
        self.ab = nn.Parameter(torch.ones((self.sizes[1] + 1, 1), dtype=self.data_type), requires_grad=True)
        self.mn = nn.Parameter(torch.ones((self.sizes[0] + 1, 1), dtype=self.data_type), requires_grad=True)
        self.act = nn.Softmax(dim=1)
        self.position_vec = nn.Embedding(self.sizes[0], self.input_size)
        self.position_vec.weight.data = self.init_size * torch.randn((self.sizes[0], self.input_size), dtype=self.data_type)
        self.rel_translation = nn.Parameter(torch.ones((self.rank, self.rank), dtype=self.data_type, requires_grad=True))
        self.common_metric = nn.Parameter(torch.ones((self.rank, self.rank), dtype=self.data_type, requires_grad=True))
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        c1 = F.softplus(self.c1[queries[:, 1]])
        head1 = expmap0(self.entity(queries[:, 0]), c1)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c1)
        rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(queries[:, 1]), lhs)
        c2 = F.softplus(self.c2[queries[:, 1]])
        head2 = expmap0(self.entity(queries[:, 0]), c2)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel11 = expmap0(rel1, c2)
        rel21 = expmap0(rel2, c2)
        lhss = project(mobius_add(head2, rel11, c2), c2)
        res11 = givens_rotations(self.rel_diag2(queries[:, 1]), lhss)
        res1 = logmap0(res1, c1)
        res11 = logmap0(res11, c2)
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, _ = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([res1.view(-1, 1, self.rank), res11.view(-1, 1, self.rank), rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])
    
    def new_get_queries(self, queries, neighbors, position):
        
        head = self.entity(queries[:, 0])
        neighbor_entity = self.entity(neighbors[queries[:, 0]][:, self.start:self.end, 0].cuda()).detach()
        neighbor_rel, _ = torch.chunk(self.rel(neighbors[queries[:, 0]][:, self.start:self.end, 1].cuda()).detach(), 2,
                                      dim=2)
        neighbor_vec = self.neighbor_vec(neighbors[queries[:, 0]][:, self.start:self.end, 1].cuda())
        att_weights = torch.sum(neighbor_vec * neighbor_rel * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_n = torch.sum(att_weights * neighbor_entity, dim=1)
        ab = torch.mean(F.softplus(self.ab[neighbors[queries[:, 0]][:, self.start:self.end, 1].cuda()]), dim=1)
        
        node_position = position[queries[:, 0]].cuda()
        position_vec = self.position_vec(queries[:, 0])
        node_position = node_position * position_vec * self.scale
        #node_position_ = node_position
        node_position = self.fc2(self.fc1(node_position.to(torch.float32)))
        node_position = self.fc3(self.norm(node_position))
        #node_position = self.ffc2(self.fc1(node_position.to(torch.float32)))
        #node_position = self.fc4(self.ffc3(self.norm2(node_position + node_position_.to(torch.float32))))
        mn = F.softplus(self.mn[queries[:, 1]])
        #
        one_hot = torch.eye(self.sizes[1])[queries[:, 1]].cuda()
        mn_ = F.softplus(self.fc_mn(one_hot))
        #
        # tmp_head = head + ab * att_n
        tmp_head = head + ab * att_n + mn_ * node_position
        #tmp_head = head + mn * node_position
        
        c1 = F.softplus(self.c1[queries[:, 1]])
        head1 = expmap0(tmp_head, c1)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c1)
        rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(queries[:, 1]), lhs)

        c2 = F.softplus(self.c2[queries[:, 1]])
        head2 = expmap0(tmp_head, c2)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel11 = expmap0(rel1, c2)
        rel21 = expmap0(rel2, c2)
        lhss = project(mobius_add(head2, rel11, c2), c2)
        res11 = givens_rotations(self.rel_diag2(queries[:, 1]), lhss)

        res1 = logmap0(res1, c1)
        res11 = logmap0(res11, c2)

        c = F.softplus(self.c[queries[:, 1]])
        rot_mat, _ = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, tmp_head).view((-1, 1, self.rank))

        cands = torch.cat([res1.view(-1, 1, self.rank), res11.view(-1, 1, self.rank), rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)

        return (res, c), self.bh(queries[:, 0])

