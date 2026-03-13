# SmartQSM - project-lightlin.github.io
# 
# Copyright (C) 2025-, YANG Jie <nj_yang_jie@foxmail.com>
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable, Any, Generator
import open3d as o3d
from scipy.spatial import Delaunay, KDTree
import networkx as nx
from utils.networkx_extra import weight_edges_by_node_data
from .pipeline import Pipeline
from scipy.special import huber
from utils.scipy_extra import berhu

class SkeletonizationPipelineBase(Pipeline):
    _edge_weight_function_to_fn: Dict[str, Callable[[Tuple[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]], float]]

    _edge_search_kwargs: Optional[Dict[str, Any]]
    _edge_weight_function: Optional[str]
    _topology_extractor: str
    
    _graph: Optional[nx.Graph]

    _points: np.ndarray
    _edges: Optional[List[Tuple[int, int]]]
    _skeletal_points: Optional[np.ndarray]
    _skeleton: Optional[nx.DiGraph]
    _radii: Optional[np.ndarray]
    _clusters: Optional[List[np.ndarray]]

    def _clear(self) -> None:
        self._edge_search_kwargs = None
        self._edge_weight_function = None
        self._topology_extractor = None

        self._edges = None
        self._graph = None
        self._clusters = None
        return 

    def __init__(
            self, 
            verbose
    ) -> None:
        super().__init__(
            verbose=verbose
        )

        self._edge_weight_function_to_fn = {
            "l1": lambda point_i_and_cluster_i, point_j_and_cluster_j : np.linalg.norm(self._skeletal_points[point_i_and_cluster_i[0]] - self._skeletal_points[point_j_and_cluster_j[0]]),
            "l2": lambda point_i_and_cluster_i, point_j_and_cluster_j: np.linalg.norm(self._skeletal_points[point_i_and_cluster_i[0]] - self._skeletal_points[point_j_and_cluster_j[0]]) ** 2
        }

        self._skeletal_points = None
        self._skeleton = None
        self._radii = None

        self._clear()
        return

    def _construct_graph(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        # Assuming there is no connectivity, use the L2 edge weighted Delaunay graph to solve for the shortest path, then expand it with existing conditions, and finally recalculate the shortest path based on the edge weight function.
        simplices: np.ndarray = Delaunay(self._skeletal_points).simplices
        # If an error occurs in this step, it indicates that the point cloud does not have sufficient points.

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
            (self._skeletal_points[start_nodes] - self._skeletal_points[end_nodes]) ** 2, #L2 only affects SPT, not MST
            axis=1
        )
        graph: nx.Graph = nx.Graph()
        graph.add_weighted_edges_from(zip(start_nodes, end_nodes, edge_weights))

        nodes: np.ndarray = np.array(list(max(nx.connected_components(graph), key=len)))
        root_node: int = nodes[np.argmin( #Fixed
            self._skeletal_points[
                nodes, # In case of graph disconnection and index modification
                2
            ]
        )]

        edges: List[Tuple[int, int]] = []
        if self._edges is not None:
            for u, v in self._edges:
                edges.append((u, v))

        node_to_predecessors: Dict[int, List[int]] = nx.dijkstra_predecessor_and_distance(
            graph,
            root_node
        )[0]

        for node, predecessors in node_to_predecessors.items():
            if len(predecessors) > 0:
                edges.append((predecessors[0], node))

        edges.extend(
            [
                (e[0], e[1]) for e in nx.minimum_spanning_edges(graph)
            ]
        )
        
        kdtree: KDTree = KDTree(self._skeletal_points)

        if "r" in self._edge_search_kwargs:
            search_radius: float = self._edge_search_kwargs["r"]
            neighbor_skeletal_point_indices_per_skeletal_point: List[np.ndarray] = kdtree.query_ball_point(
                self._skeletal_points,
                r=search_radius
            )
            for skeletal_point_idx, neighbor_skeletal_point_indices in enumerate(neighbor_skeletal_point_indices_per_skeletal_point):
                for neighbor_skeletal_point_idx in neighbor_skeletal_point_indices:
                    if neighbor_skeletal_point_idx == skeletal_point_idx:
                        continue
                    edges.append((skeletal_point_idx, neighbor_skeletal_point_idx))
        
        if "k" in self._edge_search_kwargs:
            k: int = self._edge_search_kwargs["k"]
            neighbor_skeletal_point_indices_per_skeletal_point: List[np.ndarray] = kdtree.query(
                self._skeletal_points,
                k=k
            )[1]
            for skeletal_point_idx, neighbor_skeletal_point_indices in enumerate(neighbor_skeletal_point_indices_per_skeletal_point):
                for neighbor_skeletal_point_idx in neighbor_skeletal_point_indices:
                    if neighbor_skeletal_point_idx == skeletal_point_idx:
                        continue
                    edges.append((skeletal_point_idx, neighbor_skeletal_point_idx))
        
        self._edges = np.unique(np.sort(np.array(edges), axis=1), axis=0).tolist()

        self._graph = nx.Graph()
        node_to_point_and_cluster: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        for i in range(len(self._skeletal_points)):
            node_to_point_and_cluster[i] = (
                i, 
                self._clusters[i] if self._clusters is not None else None 
            )

        weighted_edges: List[Tuple[int, int, Dict[str, float]]] = weight_edges_by_node_data(
            self._edges, 
            node_to_point_and_cluster, 
            self._edge_weight_function_to_fn[self._edge_weight_function]
        )
        
        self._graph.add_edges_from(weighted_edges)
        
        if not self._verbose:
            return
        
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])
        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(self._edges)
        lineset.colors = o3d.utility.Vector3dVector(
            np.tile([0.5, 0.5, 0.5], (len(self._edges), 1))
        )
        return f"Generated a graph with {len(self._edges)} edge(s) for skeleton extaction.", cloud, lineset
    
    def _extract_skeleton(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        nodes: np.ndarray = np.array(list(max(nx.connected_components(self._graph), key=len)))
        root_node: int = nodes[np.argmin( #Fixed
            self._skeletal_points[
                nodes, # In case of graph disconnection and index modification
                2
            ]
        )]

        # Produce skeleton
        if self._topology_extractor == "spt":
            self._skeleton = nx.DiGraph()
            node_to_predecessors: Dict[int, List[int]] = nx.dijkstra_predecessor_and_distance(
                self._graph,
                root_node
            )[0]

            for node, predecessors in node_to_predecessors.items():
                if len(predecessors) == 0:
                    continue

                self._skeleton.add_edge(
                    predecessors[0], 
                    node, 
                    weight=np.linalg.norm(self._skeletal_points[predecessors[0]] - self._skeletal_points[node])
                )
        elif self._topology_extractor == "mst":
            mst: nx.Graph = nx.minimum_spanning_tree(self._graph)
            self._skeleton = nx.dfs_tree(mst, source=root_node)
            for u, v in self._skeleton.edges:
                self._skeleton[u][v]["weight"] = np.linalg.norm(self._skeletal_points[u] - self._skeletal_points[v])
        else:
            raise NotImplementedError(f"Topology extractor {self._topology_extractor} is not implemented.")
        
        if not self._verbose:
            return
        
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector([(u, v) for u, v in self._skeleton.edges])
        lineset.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.0, 0.0], (len(self._skeleton.edges), 1))
        )

        return f"Extracted the skeleton ({self._skeleton.number_of_edges()} edges).", cloud, lineset

    def set_params(
            self,
            *,
            edge_search_kwargs: Dict[str, Any] = {},
            edge_weight_function: str = "l1",
            edge_weight_function_kwargs: Dict[str, Any] = {},
            topology_extractor: str = "spt"
    ) -> None:
        self._edge_search_kwargs = edge_search_kwargs
        self._edge_weight_function = edge_weight_function
        self._topology_extractor = topology_extractor

        self._edge_weight_function_to_fn.update(
            {
                "huber": lambda point_i_and_cluster_i, point_j_and_cluster_j, delta=edge_weight_function_kwargs.get("delta"): huber(
                    delta, 
                    np.linalg.norm(self._skeletal_points[point_i_and_cluster_i[0]] - self._skeletal_points[point_j_and_cluster_j[0]])
                ),
                "berhu": lambda point_i_and_cluster_i, point_j_and_cluster_j, delta=edge_weight_function_kwargs.get("delta"): berhu(
                    delta, 
                    np.linalg.norm(self._skeletal_points[point_i_and_cluster_i[0]] - self._skeletal_points[point_j_and_cluster_j[0]])
                ),
            }
        )

        super()._clear_pipeline()
        super()._add_fns_to_pipeline(len(self._pipeline), [
            self._construct_graph,
            self._extract_skeleton
        ])
        return
    
    def run(self, points: np.ndarray) -> Generator[Any, None, Tuple[np.ndarray, nx.DiGraph, np.ndarray]]:
        self._skeletal_points = None
        self._skeleton = None
        self._radii = None
        
        self._points = points
        for fn in self._pipeline:
            yield fn()

        self._clear()
        
        return self._skeletal_points, self._skeleton, self._radii
    