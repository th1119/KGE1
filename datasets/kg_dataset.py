"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl

import numpy as np
import torch
import itertools


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        self.table = {}
        self.nums = {}
        self.neighbors = {}
        # l<5:38471; l<4:36207; l<3:30874; l<2:18998; l<1:1428;
        # to_l<5:37356; to_l<4:35714; to_l<3:32806; to_l<2:26792; to_l<1:9236;
        self.neighbor_table = {}
        self.towards_nums = {}
        self.towards_neighbor = {}
        self.neighbors_addition = {}
        self.start = 0
        self.end = 5
        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2
        #self.get_table()
        # self.get_nums()
        #self.get_neighbors()
        #self.get_towards_neighbors()
        # self.find_neighbor_length()
        # self.get_neighbor_table()
        #self.get_nums()
        #self.get_towards_nums()
        #self.fix_neighbors()
        #self.addition_neighbors()
        #self.fix_towards_neighbors()
        # self.get_nums()
        # self.check_neighbor_table()
        self.neighbors, self.neighbors_addition = self.new_get_neighbor()
        # a=1/0

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))
    
    def new_get_neighbor(self, ):
        table = {}
        for split in ["train", "test", "valid"]:
            table[split] = torch.zeros([self.n_entities, self.n_entities], dtype=torch.int64)
            for i in self.data[split]:
                table[split][i[0]][i[2]] = i[1]
        neighbors = {}
        sample_num = self.end - self.start
        for split in ["train", "test", "valid"]:
            index_rel = []
            for i in range(self.n_entities):
                nonzero_indices = torch.nonzero(table[split][i])
                if nonzero_indices.size(0) >= sample_num:
                    random_indices = torch.randperm(nonzero_indices.numel())
                    unique_elements = nonzero_indices[random_indices[:sample_num]]
                    non_zero_rel = table[split][i][unique_elements]
                    index_rel.append(torch.cat([unique_elements, non_zero_rel], dim=1))
                else:
                    non_zero_rel = table[split][i][nonzero_indices]
                    tmp_list = torch.tensor([[i, self.n_predicates] for _ in range(sample_num - nonzero_indices.size(0))])
                    index_rel.append(torch.cat([torch.cat([nonzero_indices, non_zero_rel], dim=1), tmp_list], dim=0))
            neighbors[split] = torch.stack(index_rel, dim=0)
        neighbors_addition = {}
        for split in ["train", "test", "valid"]:
            tmp_node = neighbors[split][:, self.start:self.end, 0]
            f = lambda x: torch.tensor([_ for _ in itertools.permutations(x)])
            tmp = [f(_) for _ in tmp_node]
            tmp_ = [torch.unsqueeze(torch.cat([i for i in _], dim=0), dim=0) for _ in tmp]
            neighbors_addition[split] = torch.cat(tmp_, dim=0)
            
        return neighbors, neighbors_addition

    def get_table(self, ):
        for split in ["train", "test", "valid"]:
            self.table[split] = torch.zeros([self.n_entities, self.n_entities], dtype=torch.int64)
            for i in self.data[split]:
                self.table[split][i[0]][i[2]] = i[1]

    def get_neighbor_table(self, ):
        for split in ["train", "test", "valid"]:
            self.neighbor_table[split] = torch.full([self.n_entities, self.n_entities], self.n_predicates, dtype=torch.int64)
            for i in self.data[split]:
                self.neighbor_table[split][i[0]][i[2]] = i[1]

    def get_nums(self, ):
        for split in ["train", "test", "valid"]:
            length = [len(self.neighbors[split][j]) for j in range(len(self.neighbors[split]))]
            self.nums[split] = max(length)
            # self.nums[split] = np.count_nonzero(self.neighbors[split])

    def get_towards_nums(self, ):
        for split in ["train", "test", "valid"]:
            length = [len(self.towards_neighbor[split][j]) for j in range(len(self.towards_neighbor[split]))]
            self.towards_nums[split] = max(length)

    def get_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            index_rel = []
            for i_, i in enumerate(self.table[split]):
                non_zero_index = np.nonzero(i)
                non_zero_rel = i[non_zero_index]
                # index_rel.append(torch.cat([non_zero_index, non_zero_rel], dim=1))
                index_rel.append(torch.cat([non_zero_index, non_zero_rel], dim=1))
                # index_rel[i_] = torch.cat([non_zero_index, non_zero_rel], dim=1)
            # self.neighbors[split] = torch.cat(index_rel, dim=0)
            self.neighbors[split] = index_rel

    def get_towards_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            index_rel = []
            for i_, i in enumerate(self.table[split].t()):
                non_zero_index = np.nonzero(i)
                non_zero_rel = i[non_zero_index]
                index_rel.append(torch.cat([non_zero_index, non_zero_rel], dim=1))
            self.towards_neighbor[split] = index_rel

    def fix_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            for j in range(len(self.neighbors[split])):
                if len(self.neighbors[split][j]) < self.nums[split]:
                    #additional_data = torch.tensor([[j, j+self.n_predicates-1] for ij in range(self.nums[split]-len(self.neighbors[split][j]))])
                    additional_data = torch.tensor([[j, self.n_predicates] for ij in range(self.nums[split]-len(self.neighbors[split][j]))])
                    # self.neighbors[split][j] = torch.unsqueeze(torch.cat([self.neighbors[split][j], additional_data], dim=0), dim=0)
                    self.neighbors[split][j] = torch.cat([self.neighbors[split][j], additional_data], dim=0)
            self.neighbors[split] = torch.tensor([n.tolist() for n in self.neighbors[split]])

    def fix_towards_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            for j in range(len(self.towards_neighbor[split])):
                if len(self.towards_neighbor[split][j]) < self.towards_nums[split]:
                    additional_data = torch.tensor([[j, self.n_predicates] for _ in range(self.towards_nums[split]-len(self.towards_neighbor[split][j]))])
                    self.towards_neighbor[split][j] = torch.cat([self.towards_neighbor[split][j], additional_data], dim=0)
            self.towards_neighbor[split] = torch.tensor([n.tolist() for n in self.towards_neighbor[split]])
            
    def addition_neighbors(self, ):
        for split in ["train", "test", "valid"]:
            tmp_node = self.neighbors[split][:, self.start:self.end, 0]
            f = lambda x: torch.tensor([_ for _ in itertools.permutations(x)])
            tmp = [f(_) for _ in tmp_node]
            tmp_ = [torch.unsqueeze(torch.cat([i for i in _], dim=0), dim=0) for _ in tmp]
            self.neighbors_addition[split] = torch.cat(tmp_, dim=0)

    def check_neighbors(self, ):
        for i_, i in enumerate(self.table["train"]):
            non_zero_index = np.nonzero(i)
            print(non_zero_index[0])
            break

    def check_neighbor_table(self, ):
        cnt = 0
        for i_, i in enumerate(self.neighbor_table["train"]):
            non_zero_index = np.nonzero(i)
            if len(non_zero_index) != self.n_entities:
                cnt += 1
        print(cnt)

    def find_neighbor_length(self, ):
        cnt = 0
        for split in ["train"]:
            length = [len(self.neighbors[split][j]) for j in range(len(self.neighbors[split]))]
            for i in length:
                if i < 1:
                    cnt += 1
        print(cnt)
        a = 1/0

    def find_towards_neighbor_length(self, ):
        cnt = 0
        for split in ["train"]:
            length = [len(self.towards_neighbor[split][j]) for j in range(len(self.towards_neighbor[split]))]
            for i in length:
                if i < 1:
                    cnt += 1
        print(cnt)
        a = 1/0

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
