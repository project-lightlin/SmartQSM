import networkx as nx
import numpy as np
from typing import Callable, Any, Dict, List, Tuple, Union, Set
from scipy.spatial import KDTree, Delaunay
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
    
    def _decompose_recursively(self, node: int, path_idxx: int) -> None:
        self._paths[path_idxx].append(node)
        successors: List[int] = list(self._G.successors(node))
        if len(successors) == 1:
            self._decompose_recursively(successors[0], path_idxx)
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
            self._decompose_recursively(sorted_successors[0], path_idxx)
            for successor in sorted_successors[1:]:
                new_path_idxx: int = len(self._paths)
                self._paths.append([node])
                self._decompose_recursively(successor, new_path_idxx)
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

# Patch size is a hyperparameter required to reduce the computational complexity of a graph. Neighborhood size constrains the reachable range of each node and must be greater than the patch size.
def construct_rough_geodetic_graph_3d(
        points: np.ndarray, 
        source_selection_criteria: Callable[[np.ndarray], int] = lambda P: np.argmin(P[:, 2]), 
        max_patch_size: float = 0., 
        neighborhood_size: float = 0.
) -> Tuple[nx.DiGraph, np.ndarray, np.ndarray, List[np.ndarray]]:
    # Patchify
    # If a point cloud graph is directly generated, both k-nearest neighbors and Delaunay tetrahedration will be very slow and memory-consuming. 
    # Using the strategy of TreeQSM, the point cloud is divided into small patches, and the points within these small patches are sufficiently close to each other.
    patch_indices_of_point: np.ndarray = -np.ones(len(points), dtype=int)
    distances_to_nearest_patch_center: np.ndarray = np.full(len(points), fill_value=np.inf)
    patch_center_indices: Union[List[int], np.ndarray] = []
    kdtree: KDTree = KDTree(points)

    patch_connections: List[Tuple[int, int]] = []

    for point_idx in np.random.permutation(len(points)):
        if patch_indices_of_point[point_idx] != -1:
            continue

        patch_idx: int = len(patch_center_indices) 
        patch_center_indices.append(point_idx)

        neighbor_point_indices: np.ndarray = np.array(
            kdtree.query_ball_point(points[point_idx], r=neighborhood_size)
        )

        distances_to_patch_center: np.ndarray = np.linalg.norm(points[neighbor_point_indices] - points[point_idx], axis=1)
            
        neighbor_patch_indices: np.ndarray = np.unique(patch_indices_of_point[neighbor_point_indices])
        neighbor_patch_indices = neighbor_patch_indices[neighbor_patch_indices != -1]

        patch_connections.extend([(patch_idx, neighbor_patch_idx) for neighbor_patch_idx in neighbor_patch_indices])
        
        mask: np.ndarray = distances_to_patch_center < max_patch_size
        point_indices_in_patch: np.ndarray = neighbor_point_indices[mask]
        mask2: np.ndarray = distances_to_patch_center[mask] < distances_to_nearest_patch_center[point_indices_in_patch]
        
        point_indices_in_patch = point_indices_in_patch[mask2]
        distances_to_patch_center = distances_to_patch_center[mask][mask2]

        distances_to_nearest_patch_center[point_indices_in_patch] = distances_to_patch_center
        patch_indices_of_point[point_indices_in_patch] = patch_idx
        
    patch_center_indices = np.array(patch_center_indices)
    
    point_indices_per_patch: Union[List[Set[int]], List[np.ndarray]] = [set() for _ in range(len(patch_center_indices))]
    for point_idx, patch_idx in enumerate(patch_indices_of_point):
        point_indices_per_patch[patch_idx].add(point_idx)
    for patch_idx, point_indices in enumerate(point_indices_per_patch):
        point_indices_per_patch[patch_idx] = np.array(list(point_indices))
    
    # Make the graph connected
    patch_centers: np.ndarray = points[patch_center_indices]
    simplices: np.ndarray = Delaunay(patch_centers).simplices
    start_nodes: np.ndarray = np.concatenate([
        simplices[:, 0],
        simplices[:, 0],
        simplices[:, 0],
        simplices[:, 1],
        simplices[:, 1],
        simplices[:, 2],
    ])
    end_nodes: np.ndarray = np.concatenate([
        simplices[:, 1],
        simplices[:, 2],
        simplices[:, 3],
        simplices[:, 2],
        simplices[:, 3],
        simplices[:, 3],
    ])

    edge_weights: np.ndarray = np.linalg.norm(
        (patch_centers[start_nodes] - patch_centers[end_nodes]) ** 2, #L2 only affects SPT, not MST
        axis=1
    )

    delaunay_graph: nx.Graph = nx.Graph()
    delaunay_graph.add_weighted_edges_from(zip(start_nodes, end_nodes, edge_weights))

    nodes: np.ndarray = np.array(list(max(nx.connected_components(delaunay_graph), key=len)))
    root_node: int = nodes[np.argmin( #Fixed
        patch_centers[
            nodes, # In case of graph disconnection and index modification
            2
        ]
    )]

    node_to_predecessors: Dict[int, List[int]] = nx.dijkstra_predecessor_and_distance(
        delaunay_graph,
        root_node
    )[0]

    for node, predecessors in node_to_predecessors.items():
        if len(predecessors) > 0:
            patch_connections.append((predecessors[0], node))

    patch_connections.extend(
        [(e[0], e[1]) for e in nx.minimum_spanning_edges(delaunay_graph)]
    )    

    edges: np.ndarray = np.unique(np.sort(np.array(patch_connections), axis=1), axis=0).tolist()
    
    patch_neighborhood: nx.Graph = nx.Graph()
    patch_neighborhood.add_weighted_edges_from(
        [(u, v, np.linalg.norm(patch_centers[u] - patch_centers[v])) for u, v in edges]
    )

    # Derive geodetic graph of patches
    max_connected_components = np.array(list(max(list(nx.connected_components(patch_neighborhood)), key=len)), dtype=int)
    root_patch_idx: int = max_connected_components[source_selection_criteria(
        patch_centers[max_connected_components]
    )]

    patch_idx_to_predecessors: Dict[int, List[int]]
    patch_idx_to_shortest_distance: Dict[int, float] 
    patch_idx_to_predecessors, patch_idx_to_shortest_distance = nx.dijkstra_predecessor_and_distance(patch_neighborhood, source=root_patch_idx)
    
    # Broadcast patch-based shortest paths and max depths to each point
    shortest_distances = np.full(points.shape[0], fill_value=np.nan)

    shortest_path_edges: Union[List[np.ndarray], np.ndarray] = []
    
    for patch_idx in patch_idx_to_shortest_distance.keys():
        point_indices: np.ndarray = point_indices_per_patch[patch_idx]
        if len(patch_idx_to_predecessors[patch_idx]) == 0:
            shortest_distances[point_indices] = distances_to_nearest_patch_center[point_indices] + patch_idx_to_shortest_distance[patch_idx]
                
            shortest_path_edges.append(
                np.column_stack((np.full(len(point_indices), patch_center_indices[patch_idx], dtype=int), point_indices))
            )
            
            continue
        parent_patch_idx: int = patch_idx_to_predecessors[patch_idx][0]
        direction: np.ndarray = points[patch_center_indices[patch_idx]] - points[patch_center_indices[parent_patch_idx]]
        direction /= np.linalg.norm(direction)
        signed_distance: np.ndarray = (points[point_indices] - points[patch_center_indices[patch_idx]]) @ direction
        mask: np.ndarray = signed_distance >= 0.
        
        point_indices_for_current_patch: np.ndarray = point_indices[mask]
        point_indices_for_parent_patch: np.ndarray = point_indices[~mask]

        shortest_distances[point_indices_for_current_patch] = distances_to_nearest_patch_center[point_indices_for_current_patch] + patch_idx_to_shortest_distance[patch_idx]

        shortest_distances[point_indices_for_parent_patch] = np.linalg.norm(points[point_indices_for_parent_patch] - points[patch_center_indices[parent_patch_idx]], axis=1) + patch_idx_to_shortest_distance[parent_patch_idx]

        shortest_path_edges.append(
            np.column_stack((np.full(len(point_indices_for_current_patch), patch_center_indices[patch_idx], dtype=int), point_indices_for_current_patch))
        )
        shortest_path_edges.append(
            np.column_stack((np.full(len(point_indices_for_parent_patch), patch_center_indices[parent_patch_idx], dtype=int), point_indices_for_parent_patch))
        )

    shortest_path_edges.append(
        np.array([(patch_center_indices[us[0]], patch_center_indices[v]) for v, us in patch_idx_to_predecessors.items() if len(us) != 0])
    )
    shortest_path_edges = np.concatenate(shortest_path_edges, axis=0)
    shortest_path_edges = shortest_path_edges[shortest_path_edges[:, 0] != shortest_path_edges[:, 1]]

    geodetic_graph: nx.DiGraph = nx.DiGraph()
    geodetic_graph.add_weighted_edges_from([
        (edge[0], edge[1], np.linalg.norm(points[edge[0]] - points[edge[1]])) for edge in shortest_path_edges
    ])

    return geodetic_graph, shortest_distances, patch_center_indices, point_indices_per_patch

def is_skeleton_source_correct(T: nx.DiGraph, P: np.ndarray, criteria: Callable = lambda P: np.argmin(P[:, 2])) -> bool:
    assert is_rooted_tree(T)
    nodes: np.ndarray = list(T.nodes)
    id: int = criteria(P[nodes])
    sources = [node for node, degree in T.in_degree() if degree == 0]
    assert len(sources) == 1
    return nodes[id] == sources[0]
