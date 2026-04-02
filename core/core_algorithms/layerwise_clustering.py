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

from .core_algorithm_base import CoreAlgorithmBase
from typing import Dict, Tuple, List, Optional, Callable, Any, Set, Union, Deque
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from utils.get_distinct_colors import get_distinct_colors
from utils.networkx_extra import construct_rough_geodetic_graph_3d, MaxDepth
from collections import deque, defaultdict
from utils.clustering import Clustering

class LayerwiseClustering(CoreAlgorithmBase):
    # input
    _points: np.ndarray

    _max_patch_size: float
    _neighborhood_size: float 
    _min_layer_height: float
    _max_layer_height: Optional[float]
    _rough_height_diameter_ratio: float
    _cross_layer_cluster_spacing: float
    _stride_fn: Callable[[float, float, float], float]
    _clustering: Callable[..., List[np.ndarray]]
    _flexible: bool
    _min_cluster_size: int

    _clusters: Optional[List[np.ndarray]]
    _edges: Optional[np.ndarray]

    def __init__(
            self, 
            *, 
            verbose: bool = False, 
            max_patch_size: float = 0.04, 
            neighborhood_size:float = 0.055, 
            min_layer_height: float = 0.01, 
            max_layer_height: Optional[float] = None, 
            rough_height_diameter_ratio: float = 100, 
            stride_function: Union[Callable[[float, float, float], float], str] = lambda x, lb, ub: lb + x * (ub - lb),
            cross_layer_cluster_spacing: float = 0.04,
            flexible: bool = False,
            clustering_algorithm: str = "dbscan", 
            clustering_algorithm_kwargs: Dict[str, float] = {},
            min_cluster_size: int = 5
    ) -> None:
        super().__init__(verbose=verbose)
        self._max_patch_size = max_patch_size
        self._neighborhood_size = neighborhood_size
        self._min_layer_height = min_layer_height
        self._max_layer_height = max_layer_height
        self._rough_height_diameter_ratio = rough_height_diameter_ratio
        self._cross_layer_cluster_spacing = cross_layer_cluster_spacing
        self._flexible = flexible
        self._min_cluster_size = min_cluster_size

        if isinstance(stride_function, str):
            self._stride_fn = eval(stride_function)
        else:
            self._stride_fn = stride_function
            
        self._clear()

        self._clustering = Clustering(clustering_algorithm, **clustering_algorithm_kwargs)
        return
    
    def _clear(self) -> None:
        self._clusters = None
        self._edges = None
        return
    
    def get_pipeline(self):
        return [
            self._construct_geodetic_graph,
            self._layer_flexibly if self._flexible else self._layer_inflexibly
        ]

    def _construct_geodetic_graph(self) -> Optional[Tuple[str, o3d.geometry.LineSet]]:
        self._geodetic_graph, self._shortest_distances, _ = construct_rough_geodetic_graph_3d(self._points, max_patch_size=self._max_patch_size, neighborhood_size=self._neighborhood_size)
        
        if not self._verbose:
            return
        
        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._points)
        lineset.lines = o3d.utility.Vector2iVector(self._geodetic_graph.edges)
        lineset.paint_uniform_color([0.0, 0.0, 0.0])

        return f"Constructed the geodetic graph (max depth={np.max(self._shortest_distances)}).", lineset

    def _layer_inflexibly(self) -> Optional[Tuple[str, o3d.geometry.PointCloud]]:
        total_max_depth: float = np.max(self._shortest_distances) # Max depth = max(shortest distance)
        
        self._clusters = []
        self._edges = []
        cluster_indices_of_point: np.ndarray = -np.ones(len(self._points), dtype=int)

        layer_height: float = 0.
        previous_layer_height: float = 0.
        while True:
            if layer_height > total_max_depth:
                break

            stride: float = self._stride_fn(
                1.0 - layer_height / total_max_depth, 
                self._min_layer_height, 
                self._max_layer_height
            )

            within_layer_point_indices: np.ndarray = np.where(
                (self._shortest_distances >= layer_height) \
                & (self._shortest_distances < layer_height + stride)
            )[0]
            
            if len(within_layer_point_indices) == 0:
                previous_layer_height = layer_height
                layer_height += stride
                continue

            clusters: List[np.ndarray] = self._clustering(self._points[within_layer_point_indices])
            clusters = [within_layer_point_indices[cluster] for cluster in clusters]

            for i in reversed(range(len(clusters))):
                if len(clusters[i]) < self._min_cluster_size:
                    del clusters[i]
            
            start_cluster_idx: int = len(self._clusters)
            for cluster_idx, cluster in enumerate(clusters):
                cluster_indices_of_point[cluster] = start_cluster_idx + cluster_idx
                self._clusters.append(cluster)

            if layer_height > 0.0:
                previous_point_indices: np.ndarray = np.where(
                    (self._shortest_distances >= previous_layer_height) \
                    & (self._shortest_distances < layer_height)
                )[0]
                bilayer_point_indices: np.ndarray = np.concatenate((within_layer_point_indices, previous_point_indices))
                bilayer_clusters: List[np.ndarray] = self._clustering(self._points[bilayer_point_indices])

                bilayer_clusters = [bilayer_point_indices[cluster] for cluster in bilayer_clusters]
                for bilayer_cluster in bilayer_clusters:
                    cluster_indices: np.ndarray = np.unique(cluster_indices_of_point[bilayer_cluster])
                    cluster_indices = cluster_indices[cluster_indices != -1]
                    cluster_indices_in_current_layer: np.ndarray = cluster_indices[cluster_indices >= start_cluster_idx]
                    cluster_indices_in_previous_layer: np.ndarray = cluster_indices[cluster_indices < start_cluster_idx]
                    for cluster_idx_in_current_layer in cluster_indices_in_current_layer:
                        kdtree: KDTree = KDTree(
                            self._points[self._clusters[cluster_idx_in_current_layer]]
                        )
                        for cluster_idx_in_previous_layer in cluster_indices_in_previous_layer:
                            cluster_spacings: float = kdtree.query(
                                self._points[self._clusters[cluster_idx_in_previous_layer]], 
                                k=1
                            )[0]
                            if np.min(cluster_spacings) <= self._cross_layer_cluster_spacing:
                                self._edges.append((cluster_idx_in_previous_layer, cluster_idx_in_current_layer))
            
            previous_layer_height = layer_height
            layer_height += stride

        if not self._verbose:
            return
        cluster_colors: np.ndarray = np.vstack([np.array(get_distinct_colors(len(self._clusters) - 1)), [0.5, 0.5, 0.5]])
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._points)
        cloud.colors = o3d.utility.Vector3dVector(cluster_colors[cluster_indices_of_point])
        return f"Hierarchically clustered points into {len(self._clusters)} segments and established {len(self._edges)} pairs of connections.", cloud

    def _layer_flexibly(self) -> Optional[Tuple[str, o3d.geometry.PointCloud]]: # Efficiency improvement after optimization with Gemini3.1 Pro
        node_to_max_depth: Dict[int, float] = MaxDepth(self._geodetic_graph).compute_all()
        max_depths: np.ndarray = np.zeros(len(self._points))
        
        nodes = np.fromiter(node_to_max_depth.keys(), dtype=int, count=len(node_to_max_depth))
        depths = np.fromiter(node_to_max_depth.values(), dtype=float, count=len(node_to_max_depth))
        max_depths[nodes] = depths
        
        total_max_depth: float = max_depths.max()

        self._clusters = []
        self._edges = []

        after_allocation: np.ndarray = np.zeros(len(self._points), dtype=bool)
        after_check: np.ndarray = np.zeros(len(self._points), dtype=bool)
        
        sorted_point_indices: np.ndarray = np.argsort(max_depths)[::-1]
        cursor: int = 0

        while cursor < len(sorted_point_indices):
            point_idx: int = sorted_point_indices[cursor]
            if after_check[point_idx]:
                cursor += 1
                continue
            
            q: Deque[Tuple[np.ndarray, int]] = deque()
            q.appendleft((np.array([point_idx]), -1))

            while len(q):
                start_point_indices: np.ndarray
                parent_cluster_idx: int
                start_point_indices, parent_cluster_idx = q.popleft()
                
                current_max_depths: np.ndarray = max_depths[start_point_indices]
                key_point_idx: int = start_point_indices[np.argmax(current_max_depths)]

                max_depth: float = max_depths[key_point_idx]
                stride: float = self._stride_fn(
                    max_depth / total_max_depth, 
                    self._min_layer_height, 
                    self._max_layer_height
                )
                shortest_distance_lb: float = self._shortest_distances[key_point_idx]
                shortest_distance_ub: float = shortest_distance_lb + stride

                within_layer_point_idx_set: Set[int] = set()
                queue: deque = deque()
                cross_layer_adj = defaultdict(list)

                for point_idx in start_point_indices:
                    if self._shortest_distances[point_idx] < shortest_distance_ub: #Do not limit the lower bound!
                        queue.append(point_idx)
                        within_layer_point_idx_set.add(point_idx) 
                
                while queue:
                    point_idx: int = queue.pop()
                    for successor in self._geodetic_graph.successors(point_idx):
                        if self._shortest_distances[successor] < shortest_distance_ub: #Do not limit the lower bound!
                            if successor not in within_layer_point_idx_set:
                                queue.append(successor)
                                within_layer_point_idx_set.add(successor)
                        else:
                            cross_layer_adj[point_idx].append(successor)
                
                if not within_layer_point_idx_set:
                    continue

                within_layer_point_indices = np.fromiter(within_layer_point_idx_set, dtype=int, count=len(within_layer_point_idx_set))
                
                
                # Cluster cross_layer points
                clusters: List[np.ndarray] = self._clustering(self._points[within_layer_point_indices])

                mask: np.ndarray = np.zeros(len(within_layer_point_indices), dtype=bool)
                for cluster in clusters:
                    mask[cluster] = True

                clusters = [within_layer_point_indices[cluster] for cluster in clusters]
                
                # Although outliers are not part of the cluster, they can still be used as a starting point
                for node in within_layer_point_indices[mask == False]:
                    if after_check[node]:
                        continue
                    q.append((np.array([node]), -1))

                for i in reversed(range(len(clusters))):
                    if len(clusters[i]) < self._min_cluster_size:
                        if not np.all(after_check[clusters[i]]):
                            q.append((np.array(clusters[i]), -1))
                        del clusters[i]
                        
                after_check[within_layer_point_indices] = True
                
                # Each cluster finds its parent cluster and generates potential successors
                if not clusters:
                    continue
                
                kdtree: Optional[KDTree] = KDTree(self._points[self._clusters[parent_cluster_idx]]) if parent_cluster_idx != -1 else None
                
                for cluster in reversed(clusters):
                    cluster_idx: int = len(self._clusters)
                    mask: np.ndarray = after_allocation[cluster] == False
                    unallocated_cluster: np.ndarray = cluster[mask]
                    
                    if len(unallocated_cluster) == 0:
                        continue
                    elif len(unallocated_cluster) < self._min_cluster_size:
                        if not np.all(after_check[unallocated_cluster]):
                            q.append((unallocated_cluster, -1))
                        continue

                    after_allocation[unallocated_cluster] = True
                    self._clusters.append(unallocated_cluster)
                    
                    if kdtree:
                        cluster_spacings: np.ndarray = kdtree.query(self._points[cluster], k=1)[0]
                        if np.min(cluster_spacings) <= self._cross_layer_cluster_spacing:
                            self._edges.append((parent_cluster_idx, cluster_idx))
                    
                    if cross_layer_adj:
                        next_start_point_indices_list = []
                        for point in cluster:
                            if point in cross_layer_adj:
                                next_start_point_indices_list.extend(cross_layer_adj[point])
                        
                        if next_start_point_indices_list:
                            next_start_point_indices = np.unique(next_start_point_indices_list)
                            q.appendleft((next_start_point_indices, cluster_idx))

        if not self._verbose:
            return
        
        cluster_indices_of_point: np.ndarray = -np.ones(len(self._points), dtype=int)
        for cluster_idx, cluster in enumerate(self._clusters):
            cluster_indices_of_point[cluster] = cluster_idx

        cluster_colors: np.ndarray = np.vstack([np.array(get_distinct_colors(len(self._clusters) - 1)), [0.5, 0.5, 0.5]])
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._points)
        cloud.colors = o3d.utility.Vector3dVector(cluster_colors[cluster_indices_of_point])
        return f"Flexibly clustered points into {len(self._clusters)} segments and established {len(self._edges)} pairs of connections.", cloud
    
    def input(self, **kwargs) -> None:
        super().input(**kwargs)

        # Update member fields
        if self._max_layer_height is None:
            tree_height: float = np.max(self._points[:, 2]) - np.min(self._points[:, 2])
            self._max_layer_height = max(
                tree_height / self._rough_height_diameter_ratio * np.pi / 2.0, # Half of the circumference
                self._min_layer_height
            )
        else:
            self._max_layer_height = max(self._max_layer_height, self._min_layer_height)
        return

    def output(self):
        return {"clusters": self._clusters, "edges": self._edges}