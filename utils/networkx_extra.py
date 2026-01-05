import networkx as nx
import numpy as np
from typing import Callable, Any, Dict, List, Tuple

import sys
sys.setrecursionlimit(np.iinfo(np.int32).max)

def is_rooted_tree(G: nx.Graph) -> bool:
    if G.is_multigraph():
        return False
    if not G.is_directed():
        return False
    return (nx.is_weakly_connected(G) and 
            G.number_of_nodes() == G.number_of_edges() + 1)

class WeightedPathLength:
    _G : nx.DiGraph
    _node_to_weighted_path_length: Dict[int, float]
    _edge_attribute: str

    def __init__(self, G: nx.Graph, edge_attribute: str = "weight") -> None:
        if not is_rooted_tree(G):
            raise ValueError("Not a rooted tree.")
        
        self._G = G
        self._node_to_weighted_path_length = {
            node: 0. for node in G.nodes
        }
        self._edge_attribute = edge_attribute
        return
    
    def _compute_recursively_from_node(self, u: int) -> None:
        for v in list(self._G.successors(u)):
            self._compute_recursively_from_node(v)
            self._node_to_weighted_path_length[u] += self._node_to_weighted_path_length[v] + self._G[u][v][self._edge_attribute]
        return
    
    def _reset(self) -> None:
        self._node_to_weighted_path_length = {key: 0. for key in self._node_to_weighted_path_length}
        return
    
    def compute_one(self, node: int) -> float:
        self._reset()
        self._compute_recursively_from_node(node)
        return self._node_to_weighted_path_length[node]
    
    def compute_all(self) -> Dict[int, float]:
        self._reset()
        root_node: int
        for node, in_degree in self._G.in_degree():
            if in_degree == 0:
                root_node = node
                break
        self._compute_recursively_from_node(root_node)
        return self._node_to_weighted_path_length
    
    def compute_root(self) -> float:
        self._reset()
        root_node: int
        for node, in_degree in self._G.in_degree():
            if in_degree == 0:
                root_node = node
                break
        self._compute_recursively_from_node(root_node)
        return self._node_to_weighted_path_length[root_node]

class MaxDepth:
    _G: nx.DiGraph

    _node_to_max_depth: Dict[int, float]
    _edge_attribute: str

    def __init__(self, G: nx.Graph, edge_attribute: str = "weight") -> None:
        if not is_rooted_tree(G):
            raise ValueError("Not a rooted tree.")
        
        self._G = G

        self._node_to_max_depth = {
            node: 0. for node in G.nodes
        }
        self._edge_attribute = edge_attribute
        return
    
    def _reset(self) -> None:
        self._node_to_max_depth = {key: 0. for key in self._node_to_max_depth}
        return
    
    def _compute_recursively_from_node(self, v: int) -> None:
        for u in self._G.predecessors(v):
            max_depth = self._node_to_max_depth[v] + self._G[u][v][self._edge_attribute]
            if max_depth > self._node_to_max_depth[u]:
                self._node_to_max_depth[u] = max_depth
                self._compute_recursively_from_node(u)
            break
    
    def compute_all(self) -> Dict[int, float]:
        self._reset()
        for node, out_degree in self._G.out_degree():
            if out_degree != 0:
                continue
            self._compute_recursively_from_node(node)
        return self._node_to_max_depth
        
class GreedyPathPartitioning:
    _G: nx.DiGraph
    _node_to_attribute_value: Dict[int, float]
    _paths: List[List[int]]
    _maximized: bool

    def __init__(self, G: nx.Graph, node_to_attribute_value: Dict[int, float], maximized: bool = True) -> None:
        if not is_rooted_tree(G):
            raise ValueError("Not a rooted tree.")
        self._G = G
        self._node_to_attribute_value = node_to_attribute_value
        self._maximized = maximized

        self._paths = [[]]
        return

    def _reset(self) -> None:
        self._paths = [[]]
        return
    
    def _decompose_recursively(self, node: int, path_idx: int) -> None:
        self._paths[path_idx].append(node)
        successors: List[int] = list(self._G.successors(node))
        if len(successors) == 1:
            self._decompose_recursively(successors[0], path_idx)
        elif len(successors) == 0:
            pass
        else:
            sorted_successors: List[int] = [x for x, _ in sorted(
                [
                    (successor, self._node_to_attribute_value[successor]) 
                    for successor in successors
                ],
                key=lambda pair: pair[1],
                reverse=self._maximized
            )]
            self._decompose_recursively(sorted_successors[0], path_idx)
            for successor in sorted_successors[1:]:
                new_path_idx: int = len(self._paths)
                self._paths.append([node])
                self._decompose_recursively(successor, new_path_idx)
        return
    
    def get_paths(self) -> List[List[int]]:
        self._reset()
        root_node: int
        for node, in_degree in self._G.in_degree():
            if in_degree == 0:
                root_node = node
                break
        self._decompose_recursively(root_node, 0)
        return self._paths

def weight_edges_by_node_data(
        edges: np.ndarray,
        node_to_data: Dict[int, Any],
        edge_weight_fn: Callable[[Any, Any], float]
) -> List[Tuple[int, int, Dict[str, float]]]:
    weighted_edges: List[Tuple[int, int, Dict[str, float]]] = []
    for edge in edges:
        u: int = edge[0]
        v: int = edge[1]
        weight: float = edge_weight_fn(
            node_to_data[u], 
            node_to_data[v]
        )
        weighted_edges.append(
            (u, v, {"weight": weight})
        )
    return weighted_edges