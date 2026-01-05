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

from typing import Dict, Tuple, List, Optional, Union, Set, Callable, Any, Unpack
import numpy as np
import networkx as nx
from scipy.spatial import KDTree, Delaunay
from collections import defaultdict
import open3d as o3d
from utils.get_distinct_colors import get_distinct_colors
from utils.networkx_extra import MaxDepth
from sklearn.cluster import *
import os
import matplotlib.cm as cm

class FastLayerwiseClustering(CoreAlgorithmBase):
    _neighborhood_size: float 
    _min_layer_height: float
    _max_layer_height: Optional[float]
    _height_diameter_ratio: float
    _max_layer_spacing: float

    def __init__(self, *, verbose: bool = False, neighborhood_size: float = 0.05, min_layer_height: float = 0.01, max_layer_height: Optional[float] = None, height_diameter_ratio: float = 300, max_layer_spacing: float = 0.05,cluster_algorithm: str = "dbscan",  **kwargs) -> None:
        super().__init__(verbose=verbose)
        self._neighborhood_size = neighborhood_size
        self._min_layer_height = min_layer_height
        self._max_layer_height = max_layer_height
        self._height_diameter_ratio = height_diameter_ratio
        self._max_layer_spacing = max_layer_spacing

        self._initialize_cluster_algorithm(cluster_algorithm, kwargs)
        return
    
    def get_pipeline(self):
        return [
            self._patchify,
            self._derive_geodetic_graph,
            self._layerwise_cluster,
        ]
    
    def _initialize_cluster_algorithm(self, cluster_algorithm: str, params_for_cluster_algorithm: Dict[str, float]) -> None:
        builtin_fn: Callable[[np.ndarray, float, Any], List[np.ndarray]]
        if cluster_algorithm == "dbscan":
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            def builtin_fn(points: np.ndarray, layer_height: float, eps: float = 0.05, min_points: int = 5, **kwargs):
                cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                labels: np.ndarray = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points))
                return [np.where(labels == label)[0] for label in np.unique(labels) if label != -1]
        else:
            raise NotImplementedError(f"Cluster algorithm {cluster_algorithm} is not implemented.")
        self._clustering = lambda points, layer_height, kwargs=params_for_cluster_algorithm: builtin_fn(points, layer_height, **kwargs)
        return

    
    def _patchify(self) -> Optional[Tuple[str, o3d.geometry.PointCloud]]:
        # If a point cloud graph is directly generated, both k-nearest neighbors and Delaunay tetrahedration will be very slow and memory-consuming. 
        # Using the strategy of TreeQSM, the point cloud is divided into small patches, and the points within these small patches are sufficiently close to each other.
        patch_ids_of_point: np.ndarray = -np.ones(len(self._points), dtype=int)
        self._distances_to_nearest_patch_center: np.ndarray = np.full(len(self._points), fill_value=np.inf)
        patch_center_ids: List[np.ndarray] = []
        kdtree: KDTree = KDTree(self._points)
        edges: Union[List[np.ndarray], np.ndarray] = []
        for point_id in np.random.permutation(len(self._points)):
            if patch_ids_of_point[point_id] != -1:
                continue

            patch_id: int = len(patch_center_ids)
            patch_center_ids.append(point_id)

            point_ids_in_patch: np.ndarray = np.array(
                kdtree.query_ball_point(self._points[point_id], r=self._neighborhood_size)
            )
            distances_to_patch_center: np.ndarray = np.linalg.norm(self._points[point_ids_in_patch] - self._points[point_id], axis=1)
            
            neighbor_patch_ids: np.ndarray = np.unique(patch_ids_of_point[point_ids_in_patch])
            neighbor_patch_ids = neighbor_patch_ids[neighbor_patch_ids != -1]
            edges.append(np.column_stack((
                np.full(len(neighbor_patch_ids), fill_value=patch_id, dtype=int),
                neighbor_patch_ids
            )))

            mask: np.ndarray = distances_to_patch_center < self._distances_to_nearest_patch_center[point_ids_in_patch]
            point_ids_in_patch = point_ids_in_patch[mask]
            distances_to_patch_center = distances_to_patch_center[mask]

            self._distances_to_nearest_patch_center[point_ids_in_patch] = distances_to_patch_center
            patch_ids_of_point[point_ids_in_patch] = patch_id
        edges = np.concatenate(edges, axis=0)
        self._patch_center_ids = np.array(patch_center_ids)
        edge_weights: np.ndarray = np.linalg.norm(self._points[self._patch_center_ids[edges[:, 0]]] - self._points[self._patch_center_ids[edges[:, 1]]], axis=1)
        self._patch_neighborhood = nx.Graph()
        self._patch_neighborhood.add_weighted_edges_from([int(u), int(v), float(w)] for (u, v), w in zip(edges, edge_weights))
        point_ids_per_patch: List[Set[int]] = [set() for _ in range(len(patch_center_ids))]
        for point_id, patch_id in enumerate(patch_ids_of_point):
            point_ids_per_patch[patch_id].add(point_id)
        self._point_ids_per_patch = [None] * len(patch_center_ids)
        for patch_id, point_ids in enumerate(point_ids_per_patch):
            self._point_ids_per_patch[patch_id] = np.array(list(point_ids))

        if not self._verbose:
            return
        patch_colors: np.ndarray = np.array(get_distinct_colors(len(patch_center_ids)))
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._points)
        cloud.colors = o3d.utility.Vector3dVector(patch_colors[patch_ids_of_point])
        return f"Divided into {len(patch_center_ids)} patch(es), where {len(list(self._patch_neighborhood.edges))} pairs are interconnected.", cloud
        
    def _derive_geodetic_graph(self) -> Optional[Tuple[str, o3d.geometry.LineSet]]:
        # Make the graph connected
        simplices: np.ndarray = Delaunay(self._points[self._patch_center_ids]).simplices
        edges: np.ndarray = np.vstack((
            simplices[:, [0, 1]],
            simplices[:, [0, 2]],
            simplices[:, [0, 3]],
            simplices[:, [1, 2]],
            simplices[:, [1, 3]],
            simplices[:, [2, 3]]
        ))
        
        edges = np.column_stack((edges.min(axis=1), edges.max(axis=1)))
        edges = np.unique(edges, axis=0)
        patch_centers: np.ndarray = self._points[self._patch_center_ids]
        edge_weights = np.linalg.norm(
            patch_centers[edges[:, 0]] - patch_centers[edges[:, 1]], 
            axis=1
        )
        
        delaunay_graph: nx.Graph = nx.Graph()
        delaunay_graph.add_weighted_edges_from(
            [(int(u), int(v), float(w)) for (u, v), w in zip(edges, edge_weights)]
        )

        self._patch_neighborhood.add_edges_from(list(nx.minimum_spanning_edges(delaunay_graph))) # Ignore actual spatial distance
        
        # Derive geodetic graph of patches
        root_patch_id: int
        root_patch_z: float = np.inf
        for patch_id in max(list(nx.connected_components(self._patch_neighborhood)), key=len):
            if patch_centers[patch_id, 2] < root_patch_z:
                root_patch_id = patch_id
                root_patch_z = patch_centers[patch_id, 2]

        patch_id_to_predecessor: Dict[int, List[int]]
        patch_id_to_shortest_distance: Dict[int, float] 
        patch_id_to_predecessor, patch_id_to_shortest_distance = nx.dijkstra_predecessor_and_distance(self._patch_neighborhood, source=root_patch_id)
        
        # The maximum depth can be combined with layer height adjustment, but currently there is no good solution
        #geodetic_graph: nx.DiGraph = nx.DiGraph()
        #geodetic_graph.add_weighted_edges_from([
        #    (us[0], v, self._patch_neighborhood[us[0]][v]["weight"]) for v, us in patch_id_to_predecessor.items() if len(us) != 0
        #])
        #patch_id_to_max_depth: Dict[int, float] = MaxDepth(geodetic_graph).compute_all()

        # Broadcast patch-based shortest paths and max depths to each point
        self._shortest_distances = np.full(self._points.shape[0], fill_value=np.nan)
        #self._max_depths = np.zeros(self._points.shape[0])

        shortest_path_edges: Union[List[np.ndarray], np.ndarray] = []

        for patch_id in patch_id_to_shortest_distance.keys():
            point_ids: np.ndarray = self._point_ids_per_patch[patch_id]
            if len(patch_id_to_predecessor[patch_id]) == 0:
                self._shortest_distances[point_ids] = self._distances_to_nearest_patch_center[point_ids] + patch_id_to_shortest_distance[patch_id]
                #self._max_depths[self._patch_center_ids[patch_id]] = patch_id_to_max_depth[patch_id]
                shortest_path_edges.append(
                    np.column_stack((np.full(len(point_ids), self._patch_center_ids[patch_id], dtype=int), point_ids))
                )
                continue
            parent_patch_id: int = patch_id_to_predecessor[patch_id][0]
            direction: np.ndarray = self._points[self._patch_center_ids[patch_id]] - self._points[self._patch_center_ids[parent_patch_id]]
            direction /= np.linalg.norm(direction)
            signed_distance: np.ndarray = (self._points[point_ids] - self._points[self._patch_center_ids[patch_id]]) @ direction
            mask: np.ndarray = signed_distance >= 0.
            
            point_ids_for_current_patch: np.ndarray = point_ids[mask]
            point_ids_for_parent_patch: np.ndarray = point_ids[~mask]

            self._shortest_distances[point_ids_for_current_patch] = self._distances_to_nearest_patch_center[point_ids_for_current_patch] + patch_id_to_shortest_distance[patch_id]

            self._shortest_distances[point_ids_for_parent_patch] = np.linalg.norm(self._points[point_ids_for_parent_patch] - self._points[self._patch_center_ids[parent_patch_id]], axis=1) + patch_id_to_shortest_distance[parent_patch_id]

            #self._max_depths[self._patch_center_ids[patch_id]] = patch_id_to_max_depth[patch_id]
            shortest_path_edges.append(
                np.column_stack((np.full(len(point_ids_for_current_patch), self._patch_center_ids[patch_id], dtype=int), point_ids_for_current_patch))
            )
            shortest_path_edges.append(
                np.column_stack((np.full(len(point_ids_for_parent_patch), self._patch_center_ids[parent_patch_id], dtype=int), point_ids_for_parent_patch))
            )

        if not self._verbose:
            return
        shortest_path_edges.append(
            np.array([(self._patch_center_ids[us[0]], self._patch_center_ids[v]) for v, us in patch_id_to_predecessor.items() if len(us) != 0])
        )
        shortest_path_edges = np.concatenate(shortest_path_edges, axis=0)
        shortest_path_edges = shortest_path_edges[shortest_path_edges[:, 0] != shortest_path_edges[:, 1]]
        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._points)
        lineset.lines = o3d.utility.Vector2iVector(shortest_path_edges)
        lineset.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.0, 0.0], (len(shortest_path_edges), 1))
        )

        return f"Derived the geodetic graph (max depth={np.max(self._shortest_distances)}).", lineset

    def _layerwise_cluster(self) -> Optional[Tuple[str, o3d.geometry.PointCloud]]:
        if self._max_layer_height is None:
            tree_height: float = np.max(self._points[:, 2]) - np.min(self._points[:, 2])
            self._max_layer_height = max(tree_height / self._height_diameter_ratio, self._min_layer_height)
        else:
            self._max_layer_height = self._max_layer_height

        num_layers: int = np.ceil(2. * (np.nanmax(self._shortest_distances) + 1e-3) / (self._min_layer_height + self._max_layer_height)).astype(int)
        layer_heights: np.ndarray = np.flip(np.linspace(
            self._min_layer_height,
            self._max_layer_height,
            num_layers
        ))
        start_layer_heights: np.ndarray = np.concatenate(([0.], np.cumsum(layer_heights)))[:-1]

        self._clusters = []
        self._edges = []
        cluster_ids_of_point: np.ndarray = -np.ones(len(self._points), dtype=int)

        for layer_id in range(len(layer_heights)):
            start_layer_height: float = start_layer_heights[layer_id]
            layer_height: float = layer_heights[layer_id]
            end_layer_height: float = start_layer_height + layer_height
            point_ids: np.ndarray = np.where((self._shortest_distances >= start_layer_height) & (self._shortest_distances < end_layer_height))[0]
            clusters: List[np.ndarray] = self._clustering(self._points[point_ids], layer_height)
            clusters = [point_ids[cluster] for cluster in clusters]
            
            start_cluster_id: int = len(self._clusters)
            for cluster_id, cluster in enumerate(clusters):
                cluster_ids_of_point[cluster] = start_cluster_id + cluster_id
                self._clusters.append(cluster)

            if layer_id >= 1:
                previous_start_layer_height: float = start_layer_heights[layer_id - 1]
                previous_point_ids: np.ndarray = np.where((self._shortest_distances >= previous_start_layer_height) & (self._shortest_distances < start_layer_height))[0]
                bilayer_point_ids: np.ndarray = np.concatenate((point_ids, previous_point_ids))
                bilayer_clusters: List[np.ndarray] = self._clustering(self._points[bilayer_point_ids], layer_height)

                bilayer_clusters = [bilayer_point_ids[cluster] for cluster in bilayer_clusters]
                for bilayer_cluster in bilayer_clusters:
                    cluster_ids: np.ndarray = np.unique(cluster_ids_of_point[bilayer_cluster])
                    cluster_ids = cluster_ids[cluster_ids != -1]
                    cluster_ids_in_current_layer: np.ndarray = cluster_ids[cluster_ids >= start_cluster_id]
                    cluster_ids_in_previous_layer: np.ndarray = cluster_ids[cluster_ids < start_cluster_id]
                    for cluster_id_in_current_layer in cluster_ids_in_current_layer:
                        kdtree: KDTree = KDTree(
                            self._points[self._clusters[cluster_id_in_current_layer]]
                        )
                        for cluster_id_in_previous_layer in cluster_ids_in_previous_layer:
                            cluster_spacings: float = kdtree.query(
                                self._points[self._clusters[cluster_id_in_previous_layer]], 
                                k=1
                            )[0]
                            if np.min(cluster_spacings) <= self._max_layer_spacing:
                                self._edges.append((cluster_id_in_previous_layer, cluster_id_in_current_layer))

        if not self._verbose:
            return
        cluster_colors: np.ndarray = np.array(get_distinct_colors(len(self._clusters)))
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._points)
        cloud.colors = o3d.utility.Vector3dVector(cluster_colors[cluster_ids_of_point])
        return f"Clustered hierarchically into {len(self._clusters)} segments and established {len(self._edges)} pairs of connections.", cloud
    
    def output(self):
        return self._clusters, self._edges

registry: Dict[str, CoreAlgorithmBase] = {
    "fast_layerwise_clustering": FastLayerwiseClustering
}