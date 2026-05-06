"""
Graph Class
:description: create topology for clients
"""

import torch
import math
import numpy as np
import torch.distributed as dist
from typing import List, Dict, Tuple, Any, Optional

def closest_factors(n):
    factor1 = 1
    factor2 = n
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factor1 = i
            factor2 = n // i
    return factor1, factor2


class Edge(object):
    def __init__(self, src, dest, weight):
        self.src = src
        self.dest = dest
        self.weight = weight


class Graph(object):

    def __init__(self, topology, world_size, avg=False, reg=1e-3):
        self.topology = topology
        self.size = world_size
        self.avg = avg
        self.reg = reg
        self.adj = np.zeros((self.size, self.size))
        self.dsm = np.zeros((self.size, self.size))
        self.neighbor = [[] for i in range(self.size)]
        self.edge_list = [[] for i in range(self.size)] # 包含全部节点的边的关系，可能会有用

        self.generate_graph() # 返回整个拓扑结构的情况，包含双随机、边的情况等等

    def generate_graph(self):

        if self.topology =='Ring':
            self.generate_adj_ring()
        elif self.topology == 'Bipar':
            self.generate_adj_bipar()
        elif self.topology == 'FC':
            self.generate_adj_fc()
        elif self.topology == 'Grid':
            self.generate_adj_grid()
        elif self.topology == 'Random':
            self.generate_adj_random()
        else:
            raise NotImplementedError
        self.dsm = self.generate_doubly_stochastic_matrix_from_adj(self.adj)
        self.add_edge()
        
    def add_edge(self): # 把边加到边表
        for src in range(self.size):
            for dst in range(self.size):
                if self.adj[src][dst] == 1:
                    self.edge_list[src].append(Edge(src,dst,self.dsm[src][dst]))

    # def add_neighbor(self): # 把邻居加到邻居表
    #     for src in range(self.size):
    #         for dst in range(self.size):
    #             if self.adj[dst] == 1:
    #                 self.edge_list[src].append(dst)

    def generate_adj_ring(self):
        for src in range(self.size):
            self.adj[src][src] = 1
            self.adj[src][self.neighbor_forward(src,1)] = 1
            self.adj[src][self.neighbor_backward(src,1)] = 1

    def generate_adj_fc(self):
        self.adj = np.ones((self.size, self.size))

    def generate_adj_bipar(self):
        # 矩阵切片
        self.adj[::2, 1::2] = 1
        self.adj[1::2, ::2] = 1
        for src in range(self.size):
            self.adj[src][src] = 1

    def generate_adj_grid(self):
        rows, cols = closest_factors(self.size)
        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                # Connect to the right neighbor
                if col < cols - 1:
                    right_neighbor = index + 1
                    self.adj[index][right_neighbor] = 1
                    self.adj[right_neighbor][index] = 1
                # Connect to the bottom neighbor
                if row < rows - 1:
                    bottom_neighbor = index + cols
                    self.adj[index][bottom_neighbor] = 1
                    self.adj[bottom_neighbor][index] = 1
        for src in range(self.size):
            self.adj[src][src] = 1

    def generate_adj_random(self):
        self.adj = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            self.adj[i, i] = 1
        for i in range(self.size):
            current_non_self_loop_degree = np.sum(self.adj[i, :]) - self.adj[i, i]
            if current_non_self_loop_degree == 0:
                possible_neighbors = [j for j in range(self.size) if j != i]
                if not possible_neighbors:
                    print(f"Warning: Node {i} could not find a distinct neighbor (size={self.size}).")
                    continue
                chosen_neighbor = np.random.choice(possible_neighbors)
                self.adj[i, chosen_neighbor] = 1
                self.adj[chosen_neighbor, i] = 1 
        for i in range(self.size):
            for j in range(i + 1, self.size): 
                if self.adj[i, j] == 0 and np.random.rand() < 0.3:
                    self.adj[i, j] = 1
                    self.adj[j, i] = 1 

    def get_neighbor_info(self, rank):  # 返回当前rank的所有邻居
        neighbor_rank = np.nonzero(self.adj[rank] > 0)[0].tolist()
        weight = self.dsm[rank][np.nonzero(self.dsm[rank] > 1e-10)].tolist()
        neighbor = {}
        for idx in range(len(neighbor_rank)):
            neighbor[neighbor_rank[idx]] = weight[idx]
        return (
            neighbor_rank,  # list
            weight,         # list
            neighbor,       # dict
            self.dsm        # np.array
        )
    
    def get_comm_group(self):
        groups, dist_groups = [], []
        p2p_groups = {idx: {} for idx in range(self.size)}
        for src in range(self.size):
            group = self.get_neighbor_info(src)[0]
            groups.append(group)
            for dst in group:
                if dst > src:
                    p2p_groups[src][dst] = dist.new_group([src,dst])
                    p2p_groups[dst][src] = dist.new_group([src,dst])
            dist_groups.append(dist.new_group(group))
        return (
            groups,       # list of list 
            dist_groups,  # list of process groups
            p2p_groups    # dict of dict 
        )
    
    def neighbor_forward(self, r, p):
        """ Helper function returns peer that is p hops ahead of r """
        return (r + p) % self.size

    def neighbor_backward(self, r, p):
        """ Helper function returns peer that is p hops behind r """
        return (r - p + self.size) % self.size
    
    def sinkhorn_knopp_with_mask(self, matrix, reg=1e-2, max_iter=1000, tol=1e-10):
        mask = matrix > 0  # Create mask for non-zero elements
        row_sum = np.ones(matrix.shape[0])
        col_sum = np.ones(matrix.shape[1])
        for _ in range(max_iter):
            matrix += reg * mask
            row_sums = np.sum(matrix, axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, where=row_sums!=0)
            matrix *= mask
            col_sums = np.sum(matrix, axis=0, keepdims=True)
            matrix = np.divide(matrix, col_sums, where=col_sums!=0)
            matrix *= mask
            if np.allclose(matrix.sum(axis=1), row_sum, atol=tol) and np.allclose(matrix.sum(axis=0), col_sum, atol=tol):
                break  
        return matrix

    def generate_average_matrix(self, adj_matrix):
        degrees = np.sum(adj_matrix, axis=1)  # 每个节点的度数
        average_matrix = np.zeros_like(adj_matrix, dtype=float)
        # 计算平均矩阵
        for i in range(adj_matrix.shape[0]):
            if degrees[i] != 0:
                average_matrix[i, :] = adj_matrix[i, :] / degrees[i]
        return average_matrix

    def generate_doubly_stochastic_matrix_from_adj(self, adj_matrix):
        random_matrix = np.random.rand(*adj_matrix.shape) * adj_matrix
        if self.avg == 1:
            doubly_stochastic_matrix = self.generate_average_matrix(adj_matrix)
        else:
            doubly_stochastic_matrix = self.sinkhorn_knopp_with_mask(random_matrix, self.reg)
        return doubly_stochastic_matrix 


