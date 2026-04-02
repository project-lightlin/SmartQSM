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

from typing import Optional, List, Tuple, Union, Callable, Dict
import numpy as np
from .core_algorithm_base import CoreAlgorithmBase
from scipy.spatial import cKDTree
from utils.networkx_extra import construct_rough_geodetic_graph_3d, MaxDepth
import open3d as o3d
import sys
sys.setrecursionlimit(np.iinfo(np.int32).max)
from utils.numpy_extra import normalize

# Improved efficiency through GPT 5.2
class _Node:
    coords: np.ndarray
    previous_direction: np.ndarray
    children: List["_Node"]
    attracted_points: np.ndarray          # point ids for this iteration (np array)
    allowing_attraction: bool
    cluster: List[int]                   # accumulated point ids
    stride: float

    def __init__(self, coords: np.ndarray, previous_direction: np.ndarray, stride: float) -> None:
        self.coords = coords
        self.previous_direction = previous_direction
        self.children = []
        self.attracted_points = np.empty((0,), dtype=np.int64)
        self.allowing_attraction = True
        self.cluster = []
        self.stride = stride
        return

    def new_node(
        self,
        step_size: float,
        max_angle: float,
        max_num_branches: int,
        points: np.ndarray,
    ) -> Optional["_Node"]:
        ids = self.attracted_points
        if ids.size == 0:
            self.allowing_attraction = False
            return None

        directions = normalize(points[ids] - self.coords)
        dots = np.clip(directions @ self.previous_direction, -1.0, 1.0)
        angles = np.arccos(dots)
        valid_angle_indices = np.where(angles <= max_angle)[0]
        
        if valid_angle_indices.size == 0:
            self.allowing_attraction = False
            self.attracted_points = np.empty((0,), dtype=np.int64)
            return None
        
        self.attracted_points = self.attracted_points[valid_angle_indices]

        direction = normalize(directions[valid_angle_indices].mean(axis=0))
        new_coords = self.coords + step_size * direction

        new_node: _Node = _Node(new_coords, direction, step_size)
        self.children.append(new_node)
        self.attracted_points = np.empty((0,), dtype=np.int64)
        if len(self.children) >= max_num_branches:
            self.allowing_attraction = False

        return new_node


class SpaceColonization(CoreAlgorithmBase):
    # input
    _points: np.ndarray
    
    _max_patch_size: float
    _neighborhood_size: float

    _min_stride: float
    _max_stride: Optional[float]
    _stride_fn: Callable[[float, float, float], float]
    _stride_multiple_for_influence_radius: float
    _stride_multiple_for_kill_radius: float

    _max_angle: float
    _max_num_branches: int

    _skeletal_points: Optional[np.ndarray]
    _clusters: Optional[List[np.ndarray]]
    _edges: Optional[List[Tuple[int, int]]]

    def __init__(
            self, 
            *, 
            verbose: bool = False, 
            max_patch_size: float = 0.04, 
            neighborhood_size:float = 0.055, 
            stride_multiple_for_influence_radius: float, 
            stride_multiple_for_kill_radius: float,
            min_stride: float, 
            max_stride: Optional[float] = None,
            stride_function: Union[Callable[[float, float, float], float], str] = lambda x, lb, ub: lb + x * (ub - lb),
            rough_height_diameter_ratio: float = 100, 
            max_angle: float = np.pi / 2, 
            max_num_branches: int = 2
    ) -> None:
        self._verbose = verbose

        self._max_patch_size = max_patch_size
        self._neighborhood_size = neighborhood_size
        
        self._stride_multiple_for_influence_radius = stride_multiple_for_influence_radius
        self._stride_multiple_for_kill_radius = stride_multiple_for_kill_radius
        self._min_stride = min_stride
        if isinstance(stride_function, str):
            self._stride_fn = eval(stride_function)
        else:
            self._stride_fn = stride_function
        self._max_stride = max_stride
        self._rough_height_diameter_ratio = rough_height_diameter_ratio

        self._max_angle = max_angle
        self._max_num_branches = max_num_branches

        self._clear()
        super().__init__(verbose=verbose)
        return

    def _clear(self) -> None:
        self._skeletal_points = None
        self._clusters = None
        self._edges = None
        return

    def input(self, **kwargs) -> None:
        super().input(**kwargs)
        
        # Update member fields
        if self._max_stride is None:
            tree_height: float = np.max(self._points[:, 2]) - np.min(self._points[:, 2])
            self._max_stride = max(
                tree_height / self._rough_height_diameter_ratio,
                self._min_stride
            )
        else:
            self._max_stride = max(self._max_stride, self._min_stride)
        return

    def output(self) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int]]]:
        return {"skeletal_points": self._skeletal_points, "clusters": self._clusters, "edges": self._edges}

    def get_pipeline(self):
        return [self._grow]

    def _grow(self) -> Optional[Tuple[str, o3d.geometry.LineSet, o3d.geometry.PointCloud]]:
        points = self._points
        n_points = len(points)
        geodetic_graph, shortest_distances, _ = construct_rough_geodetic_graph_3d(
            points,
            max_patch_size=self._max_patch_size,
            neighborhood_size=self._neighborhood_size
        )
        node_to_max_depth: Dict[int, float] = MaxDepth(geodetic_graph).compute_all()
        max_depths: np.ndarray = np.zeros(len(self._points))
        nodes = np.fromiter(node_to_max_depth.keys(), dtype=int, count=len(node_to_max_depth))
        depths = np.fromiter(node_to_max_depth.values(), dtype=float, count=len(node_to_max_depth))
        max_depths[nodes] = depths
        
        total_max_depth: float = max_depths.max()

        root_point_indices: np.ndarray = np.where(shortest_distances < self._max_stride)[0]

        root = _Node(
            coords=[
                self._points[root_point_indices, 0].mean(),
                self._points[root_point_indices, 1].mean(),
                self._points[root_point_indices, 2].min() - 1e-4,
            ],
            previous_direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            stride=self._max_stride
        )
        nodes: List[_Node] = [root]

        after_allocation = np.zeros(n_points, dtype=bool)

        # One tree for the point cloud (static)
        kdtree = cKDTree(points)

        # store edges as python list, convert at end
        self._edges = []

        # global assignment buffers reused each iteration
        best_node = np.full(n_points, -1, dtype=np.int32)
        best_dist2 = np.full(n_points, np.inf, dtype=np.float64)

        while True:
            # active nodes indices this iteration
            active_idx = np.fromiter((i for i, nd in enumerate(nodes) if nd.allowing_attraction),
                                     dtype=np.int64)

            if active_idx.size == 0:
                break
            
            active_coords = np.vstack([nodes[i].coords for i in active_idx])  # (m,3)

            max_stride = np.array([nodes[i].stride for i in active_idx]).max()  # (m,)

            # batch radius query: for each active node, list of point ids in influence radius
            neigh_lists = kdtree.query_ball_point(active_coords, r=max_stride * self._stride_multiple_for_influence_radius)

            # reset only unallocated ids (fewer writes when many are already killed)
            unalloc_ids = np.where(~after_allocation)[0]
            if unalloc_ids.size == 0:
                break
            best_node[unalloc_ids] = -1
            best_dist2[unalloc_ids] = np.inf

            any_attracted = False

            # competition: assign each candidate point to closest active node
            # (still a python loop over active nodes, but heavy math inside is vectorized)
            for j, pid_list in enumerate(neigh_lists):
                if not pid_list:
                    continue
                pids = np.asarray(pid_list, dtype=np.int64)
                pids = pids[~after_allocation[pids]]
                if pids.size == 0:
                    continue

                node_id = int(active_idx[j])
                node_coord = active_coords[j]

                d = points[pids] - node_coord
                dist2 = np.einsum("ij,ij->i", d, d)

                prev = best_dist2[pids]
                upd = dist2 < prev
                if np.any(upd):
                    sel = pids[upd]
                    best_dist2[sel] = dist2[upd]
                    best_node[sel] = node_id
                    any_attracted = True

            if not any_attracted:
                break

            # clear per-iteration attracted_points
            for nd in nodes:
                nd.attracted_points = np.empty((0,), dtype=np.int64)

            # points that found an owner this iteration
            assigned_ids = unalloc_ids[best_node[unalloc_ids] != -1]
            if assigned_ids.size == 0:
                break

            owners = best_node[assigned_ids]

            # sort by owner to group in O(P log P), P = assigned points in this iter
            order = np.argsort(owners, kind="mergesort")
            assigned_ids = assigned_ids[order]
            owners = owners[order]

            # find segments for each owner
            cut = np.flatnonzero(np.diff(owners)) + 1
            owner_groups = np.split(owners, cut)
            point_groups = np.split(assigned_ids, cut)
            # mark whether node got attraction
            got_attr = np.zeros(len(nodes), dtype=bool)

            node_to_stride = {}

            # attach groups to nodes (loop over #owners, usually far smaller than #points)
            for og, pg in zip(owner_groups, point_groups):
                nid = int(og[0])
                got_attr[nid] = True

                nodes[nid].attracted_points = pg
                # accumulate cluster ids; keep python list for final unique()
                nodes[nid].cluster.extend(pg.tolist())

                node_to_stride[nid] = 0.
            
            for nid in list(node_to_stride.keys()):
                node_to_stride[nid] = self._stride_fn(
                    max_depths[
                        nodes[nid].cluster
                    ].max() / total_max_depth,
                    self._min_stride,
                    self._max_stride,
                )

            # nodes that were active but got no points: disable
            for i in active_idx:
                if not got_attr[i]:
                    nodes[i].allowing_attraction = False

            # grow new nodes; collect new coords for batch kill
            new_nodes: List[_Node] = []
            new_coords: List[np.ndarray] = []
            base_len = len(nodes)

            strides = []
            for i, nd in enumerate(nodes):
                if nd.attracted_points.size == 0:
                    continue

                stride = node_to_stride[i]
                nn = nd.new_node(
                    step_size=stride,
                    max_angle=self._max_angle,
                    max_num_branches=self._max_num_branches,
                    points=points
                )
                if nn is None:
                    continue
                self._edges.append((i, base_len + len(new_nodes)))
                new_nodes.append(nn)
                new_coords.append(nn.coords)

                strides.append(stride)

            if not new_nodes:
                break

            nodes.extend(new_nodes)

            # batch kill around all newly created nodes
            new_coords_arr = np.asarray(new_coords)
            kill_lists = kdtree.query_ball_point(new_coords_arr, r=np.array(strides) * self._stride_multiple_for_kill_radius)
            if kill_lists is not None and len(kill_lists) > 0:
                # flatten to python list first (fast in CPython), then one numpy conversion
                flat = []
                for lst in kill_lists:
                    if lst is not None and len(lst) > 0:
                        flat.extend(lst)
                if flat:
                    kill_ids = np.unique(np.asarray(flat, dtype=np.int64))
                    after_allocation[kill_ids] = True

        # finalize skeletal points and clusters
        self._clusters = []
        self._skeletal_points = np.vstack([nd.coords for nd in nodes]).astype(np.float64, copy=False)

        cluster_indices: np.ndarray = -np.ones(len(self._points), dtype=int)
        for i, nd in reversed(list(enumerate(nodes))):
            cluster: np.ndarray = np.abs(np.asarray(nd.cluster, dtype=int))
            if len(cluster) == 0:
                continue
            mask: np.ndarray = cluster_indices[cluster] == -1
            cluster = cluster[mask]
            if len(cluster) > 0:
                cluster_indices[cluster] = i
        self._clusters = [np.where(cluster_indices == i)[0] for i in range(len(nodes))]

        if not self._verbose:
            return

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(np.asarray(self._edges, dtype=np.int32))
        lineset.paint_uniform_color([0.0, 0.0, 0.0])

        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])
        

        return f"Growth finished. Generated {len(nodes)} skeletal points.", lineset, cloud