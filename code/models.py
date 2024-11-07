from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch.nn.functional as F
import torch
import numpy as np
import itertools
from functools import reduce
from torch import nn
from hyperbolic import expmap0, project,logmap0
from tqdm import tqdm
from euclidean import givens_rotations, givens_reflection
class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            neighbors: torch.Tensor,
            position: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    # scores, _ = self.forward(these_queries)
                    scores, _ = self.forward(these_queries, neighbors, position)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

        
class GIE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(GIE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.init_size=0.001
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()
        self.act = nn.Softmax(dim=1)
        #
        self.mn_num = 22
        self.start = 0
        self.end = 5
        self.neighbor_length = self.end - self.start
        self.input_size = self.neighbor_length * reduce(lambda x, y: x * y, range(self.start + 1, self.end + 1))
        self.fc1 = nn.Linear(self.input_size, self.rank)
        self.fc2 = nn.Linear(self.rank, self.rank)
        self.fc3 = nn.Linear(self.rank, self.rank)
        self.norm = nn.LayerNorm(self.rank)
        self.fc_mn = nn.Linear(self.sizes[1], 1)
        self.fc_mn1 = nn.Linear(self.sizes[1], self.mn_num)
        f_model = lambda x: np.where(x==self.sizes[1],x+1,x)

        self.embeddings = nn.ModuleList([
            nn.Embedding(f_model(s), 2 * rank, sparse=False)
            for s in sizes[:2]
        ])
        self.embeddings1 = nn.ModuleList([
            nn.Embedding(s+1, 2 * rank, sparse=False)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings1[0].weight.data *= init_size
        self.embeddings1[1].weight.data *= init_size
        self.multi_c=1;self.data_type=torch.float32
        
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1= nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)
        #
        self.ab = nn.Parameter(torch.ones((self.sizes[1] + 1, 1), dtype=self.data_type), requires_grad=True)
        self.mn = nn.Parameter(torch.ones((self.sizes[0] + 1, 1), dtype=self.data_type), requires_grad=True)
        self.neighbor_vec = nn.Embedding(self.sizes[1] + 1, self.rank)
        self.neighbor_vec.weight.data = self.init_size * torch.randn((self.sizes[1] + 1, self.rank), dtype=self.data_type)
        self.position_vec = nn.Embedding(self.sizes[0], self.rank)
        self.position_vec.weight.data = self.init_size * torch.randn((self.sizes[0], self.input_size), dtype=self.data_type)


    def forward0(self, x, neighbors, position):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        rel1 = self.embeddings1[0](x[:, 1])
        rel2 = self.embeddings1[1](x[:, 1])
        entities = self.embeddings[0].weight
        entity1 = entities[:, :self.rank]
        entity2= entities[:, self.rank:]
        lhs_t = lhs[:, :self.rank]  , lhs[:, self.rank:]
        rel = rel[:, :self.rank] , rel[:, self.rank:]
        rhs = rhs[:, :self.rank] ,rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank]
        rel2 = rel2[:, :self.rank]
        lhs=lhs_t[0]
        c1 = F.softplus(self.c1[x[:, 1]])
        head1 = expmap0(lhs, c1)
        rel11 = expmap0(rel1, c1)
        lhs = head1
        res_c1 =logmap0(givens_rotations(rel2, lhs),c1)  
        translation1=lhs_t[1] * rel[1]
        c2 = F.softplus(self.c2[x[:, 1]])
        head2 = expmap0(lhs, c2)
        rel12 = expmap0(rel1, c2)
        lhss = head2
        res_c2 = logmap0(givens_rotations(rel2, lhss),c2)  
        translation2=lhs_t[1] * rel[0]
        c = F.softplus(self.c[x[:, 1]])
        head = lhs
        rot_q = givens_rotations(rel2, head).view((-1, 1, self.rank))
        cands = torch.cat([res_c1.view(-1, 1, self.rank),res_c2.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        return (
                       (att_q * rel[0] - translation1) @ entity1.t() +(att_q * rel[1] + translation2) @ entity2.t()
               ), [
                   (torch.sqrt(lhs_t[0] ** 2 + lhs_t[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]
    
    def forward(self, x, neighbors, position):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        rel1 = self.embeddings1[0](x[:, 1])
        rel2 = self.embeddings1[1](x[:, 1])
        entities = self.embeddings[0].weight
        entity1 = entities[:, :self.rank]
        entity2= entities[:, self.rank:]
        lhs_t = lhs[:, :self.rank]  , lhs[:, self.rank:]
        rel = rel[:, :self.rank] , rel[:, self.rank:]
        rhs = rhs[:, :self.rank] ,rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank]
        rel2 = rel2[:, :self.rank]
        lhs=lhs_t[0]
        
        neighbor_entity, _ = torch.chunk(self.embeddings[0](neighbors[x[:, 0]][:, self.start:self.end, 0].cuda()).detach(), 2, dim=2)
        neighbor_rel, _ = torch.chunk(self.embeddings[1](neighbors[x[:, 0]][:, self.start:self.end, 1].cuda()).detach(), 2, dim=2)
        neighbor_vec = self.neighbor_vec(neighbors[x[:, 0]][:, self.start:self.end, 1].cuda())
        att_weights = torch.sum(neighbor_vec * neighbor_rel * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_n = torch.sum(att_weights * neighbor_entity, dim=1)
        ab = torch.mean(F.softplus(self.ab[neighbors[x[:, 0]][:, self.start:self.end, 1].cuda()]), dim=1)
        node_position = position[x[:, 0]].cuda()
        position_vec = self.position_vec(x[:, 0])
        node_position = node_position * position_vec * self.scale
        node_position = self.fc2(self.fc1(node_position.to(torch.float32)))
        node_position = self.fc3(self.norm(node_position))
        mn = F.softplus(self.mn[x[:, 1]])
        one_hot = torch.eye(self.sizes[1])[x[:, 1]].cuda()
        mn_ = F.softplus(self.fc_mn(one_hot))
        lhs = lhs + ab * att_n + mn_ * node_position
        
        c1 = F.softplus(self.c1[x[:, 1]])
        head1 = expmap0(lhs, c1)
        rel11 = expmap0(rel1, c1)
        lhs = head1
        res_c1 =logmap0(givens_rotations(rel2, lhs),c1)  
        translation1=lhs_t[1] * rel[1]
        c2 = F.softplus(self.c2[x[:, 1]])
        head2 = expmap0(lhs, c2)
        rel12 = expmap0(rel1, c2)
        lhss = head2
        res_c2 = logmap0(givens_rotations(rel2, lhss),c2)  
        translation2=lhs_t[1] * rel[0]
        c = F.softplus(self.c[x[:, 1]])
        head = lhs
        rot_q = givens_rotations(rel2, head).view((-1, 1, self.rank))
        cands = torch.cat([res_c1.view(-1, 1, self.rank),res_c2.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        return (
                       (att_q * rel[0] - translation1) @ entity1.t() +(att_q * rel[1] + translation2) @ entity2.t()
               ), [
                   (torch.sqrt(lhs_t[0] ** 2 + lhs_t[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]
    
    def forward1(self, x, neighbors, position):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        rel1 = self.embeddings1[0](x[:, 1])
        rel2 = self.embeddings1[1](x[:, 1])
        entities = self.embeddings[0].weight
        entity1 = entities[:, :self.rank]
        entity2 = entities[:, self.rank:]
        lhs_t = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank]
        rel2 = rel2[:, :self.rank]
        lhs = lhs_t[0]

        neighbor_entity, _ = torch.chunk(self.embeddings[0](neighbors[x[:, 0]][:, self.start:self.end, 0].cuda()).detach(), 2, dim=2)
        neighbor_rel, _ = torch.chunk(self.embeddings[1](neighbors[x[:, 0]][:, self.start:self.end, 1].cuda()).detach(), 2, dim=2)
        neighbor_vec = self.neighbor_vec(neighbors[x[:, 0]][:, self.start:self.end, 1].cuda())
        att_weights = torch.sum(neighbor_vec * neighbor_rel * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_n = torch.sum(att_weights * neighbor_entity, dim=1)
        ab = torch.mean(F.softplus(self.ab[neighbors[x[:, 0]][:, self.start:self.end, 1].cuda()]), dim=1)
        node_position = position[x[:, 0]].cuda()
        position_vec = self.position_vec(x[:, 0])
        node_position = node_position * position_vec * self.scale
        node_position = self.fc2(self.fc1(node_position.to(torch.float32)))
        node_position = self.fc3(self.norm(node_position))
        mn = F.softplus(self.mn[x[:, 1]])
        tmp_head = lhs + ab * att_n + mn * node_position

        c1 = F.softplus(self.c1[x[:, 1]])
        head1 = expmap0(tmp_head, c1)
        rel11 = expmap0(rel1, c1)
        # lhs = head1
        # lhs1 = project(mobius_add(head1, rel11, c1), c1)
        # lhs1 = mobius_add(head1, rel11, c1)
        res_c1 = logmap0(givens_rotations(rel2, head1), c1)
        translation1 = lhs_t[1] * rel[1]

        c2 = F.softplus(self.c2[x[:, 1]])
        # head2 = expmap0(tmp_head, c2)
        head2 = expmap0(head1, c2)
        rel12 = expmap0(rel1, c2)
        # lhs2 = project(mobius_add(head2, rel12, c2), c2)
        # lhs2 = mobius_add(head2, rel12, c2)
        res_c2 = logmap0(givens_rotations(rel2, head2), c2)
        translation2 = lhs_t[1] * rel[0]

        c = F.softplus(self.c[x[:, 1]])
        # head = lhs
        head = tmp_head
        # rot_q = givens_rotations(rel2, head).view((-1, 1, self.rank))
        rot_q = givens_rotations(rel2, head).view((-1, 1, self.rank))
        cands = torch.cat([res_c1.view(-1, 1, self.rank), res_c2.view(-1, 1, self.rank), rot_q], dim=1)
        context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        att_q = expmap0(att_q, c)
        rel_0 = expmap0(rel[0], c)
        rel_1 = expmap0(rel[1], c)
        translation1 = lhs_t[1] * rel_1
        translation2 = lhs_t[1] * rel_0
        # res = project(mobius_add(lhs, rel, c), c)

        return (
                       # (att_q * rel[0] - translation1) @ entity1.t() + (att_q * rel[1] + translation2) @ entity2.t()
                       (att_q * rel_0 - translation1) @ entity1.t() + (att_q * rel_1 + translation2) @ entity2.t()
               ), [
                   (torch.sqrt(lhs_t[0] ** 2 + lhs_t[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]
