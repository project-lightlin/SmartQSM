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
from typing import Dict, Tuple, List, Optional, Generator, Any, Union, Callable
from scipy.stats import gamma as gamma_dist
import networkx as nx
from utils.networkx_extra import WeightedPathLength, GreedyPathPartitioning, MaxDepth
import os
from joblib import delayed
import open3d as o3d
from sklearn.isotonic import IsotonicRegression
from .data_type import Branch
from scipy.spatial import KDTree
from utils.fem_pos_deviation_osqp_interface import FemPosDeviationOsqpInterface
from utils.get_distinct_colors import get_distinct_colors
from utils.open3d_extra import create_cylinder
from scipy.optimize import minimize, Bounds
from .pipeline import Pipeline

class Refinement(Pipeline):
    _trusted_aboveground_height: Optional[float]
    _trusted_height_ratio: Optional[float]
    _min_radius_weight: Optional[float]
    _x0_on_gamma_dist: Optional[float]
    _y0_on_gamma_dist: Optional[float]
    _wpl_based_correction_fn_for_radius_weights: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    _min_radius: Optional[float]

    _occupancy_factor: Optional[float]

    _weight_fem_pos_deviation: Optional[float]
    _weight_path_length: Optional[float]
    _weight_ref_deviation: Optional[float]
    _smooth_bound: Optional[float]

    _use_max_depth_instead_of_wpl: Optional[bool]
    _use_log_length: Optional[bool]
    _radius_tolerance: Optional[float]
    _allometric_equation_without_intercept: Callable[[Union[np.ndarray, float], List[float]], Union[np.ndarray, float]]
    _allometric_init_params: Optional[List[float]]
    _allometric_param_lower_bounds: Optional[List[float]]
    _allometric_param_upper_bounds: Optional[List[float]]
    _trunk_intercept_lower_bound: Optional[float]
    _trunk_intercept_upper_bound: Optional[float]
    _init_trunk_intercept: Optional[float]
    _lambda_squared_intercept_trunk: Optional[float]
    _lambda_squared_intercept_branch_curve: Optional[Callable[[float, float], float]]
    _min_num_credible_samples: Optional[int]
    _twigs_are_thinner: Optional[bool]

    _z_min: Optional[float]
    _tree_height: Optional[float]
    _surface_points: Optional[np.ndarray]
    _skeleton: Optional[nx.DiGraph]
    _skeletal_points: Optional[np.ndarray]
    _radii: Optional[np.ndarray]

    _wpls: Optional[np.ndarray] # Weighted path lengths or growth lengths (of node i) or total branch lengths (supported by node i)
    _paths: Optional[List[List[int]]]
    _reference_radii: Optional[np.ndarray]
    _path_id_to_branch: Optional[Dict[int, Branch]]
    _branch_id_to_branch: Optional[Dict[int, Branch]]

    def _clear(self) -> None:
        self._trusted_aboveground_height = None
        self._trusted_height_ratio = None
        self._min_radius_weight = None
        self._x0_on_gamma_pdf = None
        self._y0_on_gamma_pdf = None
        self._wpl_based_correction_fn_for_radius_weights = None
        self._min_radius = None

        self._occupancy_factor = None

        self._weight_fem_pos_deviation = None
        self._weight_path_length = None
        self._weight_ref_deviation = None
        self._smooth_bound = None

        self._use_max_depth_instead_of_wpl = None
        self._use_log_length = None
        self._radius_tolerance = None
        self._allometric_equation_without_intercept = None
        self._allometric_init_params = None
        self._allometric_param_lower_bounds = None
        self._allometric_param_upper_bounds = None
        self._trunk_intercept_lower_bound = None
        self._trunk_intercept_upper_bound = None
        self._init_trunk_intercept = None
        self._lambda_squared_intercept_trunk = None
        self._lambda_squared_intercept_branch_curve = None
        self._min_num_credible_samples = None
        self._twigs_are_thinner = None

        self._z_min = None
        self._tree_height = None
        self._surface_points = None
        self._skeleton = None
        self._skeletal_points = None
        self._radii = None
        
        self._wpls = None
        self._paths = None
        self._reference_radii = None
        self._path_id_to_branch = None
        return

    def __init__(
            self, 
            *,
            verbose: bool = False
    ) -> None:
        super().__init__(
            verbose=verbose
        )
        
        self._clear()
        self._branch_id_to_branch = None

        return
    
    def set_params(
            self,
            *,
            trusted_aboveground_height: float = 1.3,
            trusted_height_ratio: float = 0.2,
            min_radius_weight: float = 1.0e-6,
            x0_on_gamma_pdf: Optional[float] = None,
            y0_on_gamma_pdf: Optional[float] = None,
            wpl_based_correction_fn_for_radius_weights: Union[Callable[[np.ndarray, np.ndarray, float], np.ndarray], str] = lambda radius_weights, wpls: radius_weights * (np.log(wpls + 1.0) / np.log(np.max(wpls) + 1.0)),
            min_radius: float = 1.0e-3,
            occupancy_factor: float = 1.1,
            weight_fem_pos_deviation: float = 1.0e10,
            weight_path_length: float = 1.0,
            weight_ref_deviation: float = 1.0,
            smooth_bound: float = 0.05,
            use_max_depth_instead_of_wpl: bool = False,
            use_log_length: bool = True,
            radius_tolerance: float = 0.1,
            allometric_equation_without_intercept: Union[Callable[[Union[np.ndarray, float], List[float]], Union[np.ndarray, float]], str] = lambda x, params: params[0] * x ** params[1],
            allometric_init_params: Optional[List[float]] = [1.0e-2, 3.0],
            allometric_param_lower_bounds: Optional[List[float]] = [1.0e-6, 1.5],
            allometric_param_upper_bounds: Optional[List[float]] = [1.0e6, 5.0],
            trunk_intercept_lower_bound: float = 0.0,
            trunk_intercept_upper_bound: float = 1.0e6,
            init_trunk_intercept: float = 0.0,
            lambda_squared_intercept_trunk: float = 0.1,
            lambda_squared_intercept_branch_curve: Union[Callable[[float, float], float], str] = lambda wpl, wpl_max: (0.1 / 10.0) ** (np.log(wpl + 1.0) / np.log(wpl_max + 1.0)) * 10.0,
            min_num_credible_samples: int = 5,
            twigs_are_thinner: bool = True
    ) -> None:
        if min_radius <= 0.:
            raise ValueError("min_radius must be > 0.")
        self._min_radius = min_radius
        if radius_tolerance < 0. or radius_tolerance > 1.:
            raise ValueError("radius_tolerance must be >= 0. and <= 1.")
        self._radius_tolerance = radius_tolerance
        if occupancy_factor <= 0.:
            raise ValueError("occupancy_factor must be > 0.")
        self._occupancy_factor = occupancy_factor
        if trusted_aboveground_height < 0.:
            raise ValueError("trusted_aboveground_height should be >= 0 (relative height [meter] to the ground).")
        self._trusted_aboveground_height = trusted_aboveground_height
        if trusted_height_ratio < 0. or trusted_height_ratio > 1.:
            raise ValueError("trusted_height_ratio must be >= 0. and <= 1.")
        self._trusted_height_ratio = trusted_height_ratio
        if min_radius_weight <= 0.:
            raise ValueError("min_radius_weight must be > 0.")
        self._min_radius_weight = min_radius_weight

        if callable(wpl_based_correction_fn_for_radius_weights):
            self._wpl_based_correction_fn_for_radius_weights = wpl_based_correction_fn_for_radius_weights
        else:
            self._wpl_based_correction_fn_for_radius_weights = eval(wpl_based_correction_fn_for_radius_weights)

        self._x0_on_gamma_pdf = x0_on_gamma_pdf
        self._y0_on_gamma_pdf = y0_on_gamma_pdf

        self._weight_fem_pos_deviation = weight_fem_pos_deviation
        self._weight_path_length = weight_path_length
        self._weight_ref_deviation = weight_ref_deviation
        self._use_max_depth_instead_of_wpl = use_max_depth_instead_of_wpl
        self._use_log_length = use_log_length
        self._smooth_bound = smooth_bound

        if callable(allometric_equation_without_intercept):
            self._allometric_equation_without_intercept = allometric_equation_without_intercept
        else:
            self._allometric_equation_without_intercept = eval(allometric_equation_without_intercept)
        self._allometric_init_params = allometric_init_params
        self._allometric_param_lower_bounds = allometric_param_lower_bounds
        self._allometric_param_upper_bounds = allometric_param_upper_bounds
        self._trunk_intercept_lower_bound = trunk_intercept_lower_bound
        self._trunk_intercept_upper_bound = trunk_intercept_upper_bound
        self._init_trunk_intercept = init_trunk_intercept

        self._lambda_squared_intercept_trunk = lambda_squared_intercept_trunk
        if callable(lambda_squared_intercept_branch_curve):
            self._lambda_squared_intercept_branch_curve = lambda_squared_intercept_branch_curve
        else:
            self._lambda_squared_intercept_branch_curve = eval(lambda_squared_intercept_branch_curve)

        if min_num_credible_samples <= 0:
            raise ValueError("min_num_credible_samples must be > 0.")
        self._min_num_credible_samples = min_num_credible_samples
        self._twigs_are_thinner = twigs_are_thinner

        super()._clear_pipeline()
        super()._add_fns_to_pipeline(len(self._pipeline), [
            self._partition_paths,
            self._generate_branches,
            self._calculate_reference_radii,
            self._identify_reasonable_branches,
            self._update_skeleton,
            self._partition_paths, # Required! As many twigs are removed, WPLs have changed and the selection of thicker branch at the furcation should be different from before.
            self._generate_branches,
            self._identify_reasonable_branches, # Required! 
            self._smooth_pathwise,
            self._update_skeleton,
            self._estimate_radii,
            self._convert_data_format
        ])
        return

    def _partition_paths(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        # Compute WPL and partition path greedlily
        # On different branch segments, there is a significant difference in the numerical values of weight path lengths
        node_to_weighted_path_length: Dict[int, float] = WeightedPathLength(self._skeleton).compute_all()
        self._wpls = np.zeros(len(self._skeletal_points), dtype=float)
        for node, weighted_path_length in node_to_weighted_path_length.items():
            self._wpls[node] = weighted_path_length
        self._paths = GreedyPathPartitioning(
            self._skeleton, 
            node_to_weighted_path_length, 
            maximized=True
        ).get_paths()

        if not self._verbose:
            return
        
        path_colors: np.ndarray = np.array(get_distinct_colors(len(self._paths)))
        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        edges: List[Tuple[int, int]] = []
        edge_colors: List[Tuple[float, float, float]] = []
        for path_id, path in enumerate(self._paths):
            for i in range(len(path) - 1):
                edges.append((path[i], path[i + 1]))
                edge_colors.append(path_colors[path_id])
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(edge_colors)
        return f"Partitioned the skeleton into {len(self._paths)} paths.", lineset

    def _generate_branches(self) -> Optional[Tuple[str, o3d.geometry.LineSet]]:
        branch_orders_of_node: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        path_ids_of_node: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        self._path_id_to_branch = {}
        for path_id, path in enumerate(self._paths):
            branch: Branch = Branch()
            parent_branch_id: int = -1
            branch_order: int = 0
            joint_point_id: int = -1
            if path_id != 0:
                parent_branch_id = path_ids_of_node[path[0]]
                branch_order = branch_orders_of_node[path[0]] + 1
                joint_point_id = path[0]
                path = path[1:] # Remove the joint point from each branch
            branch.order = branch_order
            branch.parent_id = parent_branch_id
            branch.base_radius = self._skeletal_points[path[0], 2] # Temporarily borrow this attribute to store height for sorting
            branch.joint_point_id = joint_point_id
            self._path_id_to_branch[path_id] = branch
            branch_orders_of_node[path] = branch_order
            path_ids_of_node[path] = path_id
            self._paths[path_id] = path

        # Sort dictionary in ascending sequence of order and descending sequence of height
        self._path_id_to_branch = dict(sorted(
            self._path_id_to_branch.items(), 
            key=lambda item: (item[1].order, -item[1].base_radius)
        ))

        if not self._verbose:
            return
        
        max_branch_order: int = 0
        for path_id, branch in self._path_id_to_branch.items():
            max_branch_order = max(max_branch_order, branch.order)

        branch_order_colors: List[Tuple[float, float, float]] = get_distinct_colors(max_branch_order + 1)
        
        edges: List[Tuple[int, int]] = []
        edge_colors: List[Tuple[float, float, float]] = []
        for path_id, branch in self._path_id_to_branch.items():
            path: List[int] = self._paths[path_id]
            branch_order: int = branch.order
            for j in range(len(path) - 1):
                edges.append((path[j], path[j + 1]))
                edge_colors.append(branch_order_colors[branch_order])

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(edge_colors)

        return f"Generated {len(self._path_id_to_branch)} branches (Max branch order={max_branch_order}).", lineset
    
    @staticmethod
    def estimate_gamma_dist_params(extremum: float, x0: float, y0: float) -> Tuple[float, float]:
        assert extremum > 0 and x0 > extremum
        assert 0 < y0 < 1
        scale: float = (x0 - extremum) / (-np.log(y0))
        a: float = 1.0 + extremum / scale
        return a, scale

    def _calculate_reference_radii(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        a:float
        scale:float 
        a, scale = self.estimate_gamma_dist_params(
            extremum = min(
                self._tree_height * self._trusted_height_ratio, 
                self._trusted_aboveground_height
            ), 
            x0 = self._tree_height if self._x0_on_gamma_pdf is None else self._x0_on_gamma_pdf, 
            y0 = self._min_radius_weight if self._y0_on_gamma_pdf is None else self._y0_on_gamma_pdf
        )

        weights: np.ndarray = gamma_dist(
            a=a, 
            loc=0.0, 
            scale=scale
        ).pdf(self._skeletal_points[:, 2] - self._z_min)

        weights = np.clip(
            self._wpl_based_correction_fn_for_radius_weights(weights, self._wpls), 
            self._min_radius_weight, 
            None
        ) # When weight = 0, the result of Isotonic regression could be nan
        
        self._reference_radii = np.zeros(len(self._skeletal_points))

        for path_id, branch in self._path_id_to_branch.items():
            path: List[int] = self._paths[path_id]
            if path_id != 0:
                path.insert(0, branch.joint_point_id)
            pathwise_lengths: np.ndarray = self._wpls[path]
            pathwise_radii: np.ndarray = self._radii[path].copy()
            pathwise_weights: np.ndarray = weights[path].copy()
            
            if path_id != 0:
                pathwise_radii[0] = self._reference_radii[path[0]]
                pathwise_weights[0] = 1e4

            pathwise_reference_radii: np.ndarray = IsotonicRegression().fit_transform(
                pathwise_lengths, 
                pathwise_radii, 
                sample_weight = pathwise_weights
            )
            if path_id != 0:
                self._reference_radii[path[1:]] = pathwise_reference_radii[1:]
            else:
                self._reference_radii[path] = pathwise_reference_radii

        self._reference_radii[self._reference_radii < self._min_radius] = self._min_radius

        if not self._verbose:
            return
        
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])
        
        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        for point_id, point in enumerate(self._skeletal_points):
            radius: float = self._reference_radii[point_id]
            sphere_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
            sphere_mesh.translate(point)
            sphere_mesh.compute_vertex_normals()
            mesh += sphere_mesh
        
        return f"Calculated the reference radii (max={np.max(self._reference_radii):.4f}).", mesh

    def _identify_reasonable_branches(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        # Thicker branches in the same order are given priority
        for path_id, branch in self._path_id_to_branch.items():
            branch.base_radius = self._reference_radii[self._paths[path_id][0]]
        self._path_id_to_branch = dict(sorted(
            self._path_id_to_branch.items(), 
            key=lambda item: (item[1].order, -item[1].base_radius)
        ))
         

        # Occupy nodes with a series of balls along each path
        path_ids_of_node = -np.ones(len(self._skeletal_points), dtype=int)
        checked: np.ndarray = np.zeros(len(self._skeletal_points), dtype=bool)
        kdtree: KDTree = KDTree(self._skeletal_points)
        for path_id in list(self._path_id_to_branch.keys()):
            path: List[int] = self._paths[path_id]
            parent_branch_id: int = -1
            joint_node: int = -1
            if path_id > 0:
                # Remove nodes at the furcation
                while len(path) > 0:
                    if checked[path[0]] == True: 
                        path = path[1:]
                    else:
                        break
                
                # Remove all intermediate occupied nodes
                # The remaining part maintains continuity
                checked_id: int = 0
                while checked_id < len(path):
                    if checked[path[checked_id]] == False:
                        checked_id += 1
                    else:
                        path = path[:checked_id] + path[checked_id + 1:]
                
            self._paths[path_id] = path

            if path_id > 0:
                if len(path) <= 1:
                    del self._path_id_to_branch[path_id]
                    continue

                # Find parent branch
                joint_node = list(self._skeleton.predecessors(path[0]))[0]
                while True:
                    if path_ids_of_node[joint_node] != -1 and checked[joint_node] == True:
                        parent_branch_id = path_ids_of_node[joint_node]
                        break
                    predecessors: List[int] = list(self._skeleton.predecessors(joint_node))
                    joint_node = predecessors[0] # It will definitely not be empty
                    
            self._path_id_to_branch[path_id].parent_id = parent_branch_id
            self._path_id_to_branch[path_id].joint_point_id = joint_node # For temporary use
            if joint_node != -1:
                self._path_id_to_branch[path_id].order = self._path_id_to_branch[parent_branch_id].order + 1
            
            neighbor_indices_per_node: List[List[int]] = kdtree.query_ball_point(
                self._skeletal_points[path], 
                r=self._reference_radii[path] * self._occupancy_factor
            )

            inlier_nodes: np.ndarray = np.unique(
                np.concatenate(neighbor_indices_per_node)
            )
            
            checked[inlier_nodes] = True
        
            path_ids_of_node[path] = path_id

        if not self._verbose:
            return

        path_colors: List[Tuple[float, float, float]] = get_distinct_colors(len(self._path_id_to_branch))
        
        edges: List[Tuple[int, int]] = []
        edge_colors: List[Tuple[float, float, float]] = []
        for i, path_id in enumerate(self._path_id_to_branch.keys()):
            path: List[int] = self._paths[path_id]
            for j in range(len(path) - 1):
                edges.append((path[j], path[j + 1]))
                edge_colors.append(path_colors[i])

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(edge_colors)

        return f"Identified {len(self._path_id_to_branch)} reasonable branches.", lineset
    
    def _update_skeleton(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        self._skeleton = nx.DiGraph()
        for path_id, branch in self._path_id_to_branch.items():
            path: List[int] = self._paths[path_id]
            for i in range(len(path) - 1):
                self._skeleton.add_edge(path[i], path[i + 1], weight=np.linalg.norm(self._skeletal_points[path[i]] - self._skeletal_points[path[i + 1]]))
            if branch.parent_id != -1:
                self._skeleton.add_edge(branch.joint_point_id, path[0], weight=np.linalg.norm(self._skeletal_points[branch.joint_point_id] - self._skeletal_points[path[0]]))

        if not self._verbose:
            return
        
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])
        
        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(self._skeleton.edges)
        lineset.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.0, 0.0], (len(self._skeleton.edges), 1))
        )

        return f"Updated the skeleton to {len(self._skeleton.edges)} edges.", cloud, lineset

    
    def _smooth_pathwise(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        updated_skeletal_points: np.ndarray = self._skeletal_points.copy()
        
        fem_pos_deviation_smoother: FemPosDeviationOsqpInterface = FemPosDeviationOsqpInterface()
        fem_pos_deviation_smoother.set_weight_fem_pos_deviation(self._weight_fem_pos_deviation)
        fem_pos_deviation_smoother.set_weight_path_length(self._weight_path_length)
        fem_pos_deviation_smoother.set_weight_ref_deviation(self._weight_ref_deviation)
        
        count: int = 0
        for path_id in self._path_id_to_branch.keys():
            path: List[int] = self._paths[path_id]
            if len(path) < 3:
                continue
            elif len(path) == 3:
                updated_skeletal_points[path[1]] = (self._skeletal_points[path[0]] + self._skeletal_points[path[2]]) / 2.0
                continue
            count += 1
            
            points: np.ndarray = self._skeletal_points[path]

            bounds: np.ndarray = np.full(len(points), self._smooth_bound)
            
            bounds[0] = 0.0
            bounds[-1] = 0.0

            fem_pos_deviation_smoother.set_ref_points(points)
            fem_pos_deviation_smoother.set_bounds_around_refs(bounds)
            if fem_pos_deviation_smoother.Solve():
                updated_skeletal_points[path] = fem_pos_deviation_smoother.opt_points_

        # Smoothed branching joints have shifted
        for path_id, branch in self._path_id_to_branch.items():
            parent_branch_id: int = branch.parent_id
            if parent_branch_id == -1:
                continue
            path: List[int] = self._paths[path_id]
            parent_path: List[int] = self._paths[parent_branch_id]
            kdtree: KDTree = KDTree(updated_skeletal_points[parent_path])
            joint_node: int = parent_path[kdtree.query(self._skeletal_points[branch.joint_point_id], k=1)[1]]
            self._path_id_to_branch[path_id].joint_point_id = joint_node
        
        self._skeletal_points = updated_skeletal_points

        if not self._verbose:
            return
        
        edges: List[Tuple[int, int]] = []
        for path_id in self._path_id_to_branch.keys():
            for i in range(len(self._paths[path_id]) - 1):
                edges.append(
                    (self._paths[path_id][i], self._paths[path_id][i + 1])
                )

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.0, 0.0], (len(edges), 1))
        )

        return f"Smoothed {count} long paths in the skeleton.", lineset
    
    def _estimate_radii(self) -> Optional[Tuple[str, o3d.geometry.TriangleMesh]]:
        self._wpls = np.zeros(len(self._skeletal_points), dtype=float)
        node_to_weighted_path_length: Dict[int, float] = WeightedPathLength(self._skeleton).compute_all()
        for node, weighted_path_length in node_to_weighted_path_length.items():
            self._wpls[node] = weighted_path_length
        
        stem_length: float = np.max(self._wpls)

        lengths: np.ndarray 
        if self._use_max_depth_instead_of_wpl:
            max_depths: np.ndarray = np.zeros(len(self._skeletal_points), dtype=int)
            node_to_max_depth: Dict[int, int] = MaxDepth(self._skeleton).compute_all()
            for node, max_depth in node_to_max_depth.items():
                max_depths[node] = max_depth
            lengths = max_depths.copy()
        else:
            lengths = self._wpls.copy()

        if self._use_log_length:
            lengths = np.log(lengths + 1.0)

        # Strengthen shape control
        a:float
        scale:float 
        a, scale = self.estimate_gamma_dist_params(
            extremum = min(
                self._tree_height * self._trusted_height_ratio, 
                self._trusted_aboveground_height
            ), 
            x0 = self._tree_height if self._x0_on_gamma_pdf is None else self._x0_on_gamma_pdf, 
            y0 = self._min_radius_weight if self._y0_on_gamma_pdf is None else self._y0_on_gamma_pdf
        )

        weights: np.ndarray = gamma_dist(
            a=a, 
            loc=0.0, 
            scale=scale
        ).pdf(self._skeletal_points[:, 2] - self._z_min)

        weights = np.clip(
            self._wpl_based_correction_fn_for_radius_weights(weights, self._wpls), 
            self._min_radius_weight, 
            None
        )

        weights[
            (self._radii < self._reference_radii * (1 - self._radius_tolerance))
            | (self._radii > self._reference_radii * (1 + self._radius_tolerance))
        ] = 0.0
        
        allometric_parameters: np.ndarray
        intercepts: Dict[int, float] = {}

        for path_id, branch in self._path_id_to_branch.items():
            path: List[int] = self._paths[path_id]
            if path_id == 0:
                sample_weights: np.ndarray = weights[path]

                # Fit y=ax^b+c
                # Optimize \min \sum_i^n w_i (a x_i ^ b + c - y_i)^2 / \sum_i^n w_i + lambda * c^2
                params: np.ndarray = minimize(
                    lambda params, x, y, w, lambda_squared_intercept: np.sum(w * (y - (self._allometric_equation_without_intercept(x, params[:-1]) + params[-1]))**2) / np.sum(w) + lambda_squared_intercept * params[2] ** 2,
                    x0=self._allometric_init_params + [self._init_trunk_intercept],
                    args=(lengths[path], self._radii[path], sample_weights, self._lambda_squared_intercept_trunk),
                    bounds=Bounds(
                        lb=self._allometric_param_lower_bounds + [self._trunk_intercept_lower_bound],
                        ub=self._allometric_param_upper_bounds + [self._trunk_intercept_upper_bound]
                    ),
                    method="L-BFGS-B",
                    options={
                        "maxiter": 5000,
                        "ftol": 1e-12
                    }
                ).x

                self._radii[path] = np.clip(
                    self._allometric_equation_without_intercept(lengths[path], params[:-1]) + params[-1], 
                    self._min_radius, 
                    np.inf
                )
                allometric_parameters = params[:-1]
                intercepts[path_id] = params[-1]
                self._path_id_to_branch[path_id].base_radius = np.nan
            else:
                # Given a0, b0 and c0, fit y=a0x^b0+c where initial c=c0 and c is restricted by the parent branch: 
                # The thickness of the sub branch must not exceed the subsequent thickness of the parent branch at the branching joint
                sample_weights: np.ndarray = weights[path]

                joint_node: int = branch.joint_point_id
                parent_branch_id: int = branch.parent_id
                
                intercept: float = intercepts[parent_branch_id]
                max_weighted_path_length: float = 0.
                current_weighed_path_length: float = 0.
                for successor in list(self._skeleton.successors(joint_node)):
                    weighted_path_length_of_successor: float = lengths[successor] + np.linalg.norm(self._skeletal_points[successor] - self._skeletal_points[joint_node])
                    if successor == path[0]:
                        current_weighed_path_length = weighted_path_length_of_successor
                    max_weighted_path_length = max(
                        max_weighted_path_length,
                        weighted_path_length_of_successor
                    )

                if np.sum(sample_weights != 0.0) >= self._min_num_credible_samples:
                    max_intercept = intercept + self._allometric_equation_without_intercept(
                        max_weighted_path_length, 
                        allometric_parameters
                    ) - self._allometric_equation_without_intercept(
                        current_weighed_path_length, 
                        allometric_parameters
                    )

                    lambda_squared_intercept_branch: float = self._lambda_squared_intercept_branch_curve(self._wpls[path[0]], stem_length) # This is a compromise that ensures a lighter intercept penalty for larger branches to better fit the point cloud
                    
                    params: np.ndarray = minimize(
                        fun=lambda params, x, y, w, lambda_squared_intercept, allometric_parameters=allometric_parameters: np.sum(w * (y - (self._allometric_equation_without_intercept(x, allometric_parameters) + params[0])) ** 2 ) / np.sum(w) + lambda_squared_intercept * params[0] ** 2,
                        x0=[intercept],
                        bounds=Bounds(lb=[-np.inf], ub=[max_intercept]),
                        args=(
                            lengths[path],
                            self._radii[path],
                            sample_weights,
                            lambda_squared_intercept_branch
                        ),
                        options={
                            "maxiter": 5000,
                            "ftol": 1e-12
                        }
                    ).x

                    intercept = params[0]
                else:
                    if self._twigs_are_thinner:
                        intercept = min(intercept, intercepts[0])

                intercepts[path_id] = intercept
                self._radii[path] = np.clip(self._allometric_equation_without_intercept(lengths[path], allometric_parameters) + intercept, self._min_radius, np.inf)
                self._path_id_to_branch[path_id].base_radius = max(self._allometric_equation_without_intercept(current_weighed_path_length, allometric_parameters) + intercept, self._min_radius) # radius of branching joint
                
            # Smooth radius change at the branching joint
            # Construct a sphere at each branching point on the path, 
            # and cover the nodes with the larger of the sphere's radius and the original radius
            mask: np.ndarray = np.zeros(len(path), dtype=bool)
            kdtree: KDTree = KDTree(self._skeletal_points[path])
            for i, node in enumerate(path):
                if mask[i] == True:
                    continue
                if self._skeleton.out_degree(node) <= 1:
                    continue
                mask[i] = True
                neighbor_ids: List[int] = kdtree.query_ball_point(self._skeletal_points[node], self._radii[node])
                for neighbor_id in neighbor_ids:
                    if mask[neighbor_id] == True:
                        continue
                    mask[neighbor_id] = True
                    descendant: int = path[neighbor_id]
                    
                    self._radii[descendant] = max(
                        self._radii[descendant],
                        np.sqrt(self._radii[node] ** 2 - np.linalg.norm(self._skeletal_points[node] - self._skeletal_points[descendant]) ** 2)
                    )
        
        if not self._verbose:
            return
        
        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh() 

        for path_id, branch in self._path_id_to_branch.items():
            path: List[int] = self._paths[path_id]
            for node in path:
                radius: float = self._radii[node]
                sphere_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=radius, 
                    resolution=6
                )
                sphere_mesh.translate(self._skeletal_points[node])
                sphere_mesh.compute_vertex_normals()
                mesh += sphere_mesh

        return f"Estimated the radii (max={np.max(self._radii):.4f}).", mesh
    
    def _convert_data_format(self) -> Optional[Tuple[str, o3d.geometry.TriangleMesh]]:
        # No longer using index structures in Branch data type
        indices_on_each_path: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        
        branch_id_to_branch: Dict[int, Branch] = {}
        for path_id, branch in self._path_id_to_branch.items():
            path: List[int] = self._paths[path_id]
            new_branch: Branch = Branch()
            new_branch.medial_points = self._skeletal_points[path]
            new_branch.radii = self._radii[path]
            new_branch.order = branch.order
            new_branch.base_radius = branch.base_radius # radius of branching joint
            if branch.parent_id != -1:
                new_branch.joint_point_id = indices_on_each_path[branch.joint_point_id]
                new_branch.parent_id = branch.parent_id + 1 # Match TreeQSM
            else:
                new_branch.joint_point_id = -1
                new_branch.parent_id = -1
            branch_id_to_branch[path_id + 1] = new_branch # Match TreeQSM
            indices_on_each_path[path] = np.linspace(0, len(path) - 1, len(path), dtype=int)

        self._branch_id_to_branch = branch_id_to_branch

        if not self._verbose:
            return
        
        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()

        for branch_id, branch in self._branch_id_to_branch.items():
            medial_points: np.ndarray = branch.medial_points
            radii: np.ndarray = branch.radii

            for i in range(len(medial_points) - 1):
                start_point: np.ndarray = medial_points[i]
                end_point: np.ndarray = medial_points[i + 1]
                radius: float = radii[i + 1]
                
                mesh += create_cylinder(
                    start_point=start_point,
                    end_point=end_point,
                    radius=radius,
                    resolution=12
                )
                
        return f"Converted data format.", mesh

    def run(self, surface_points: np.ndarray, skeletal_points: np.ndarray, skeleton: nx.DiGraph, radii: np.ndarray) -> Generator[Any, None, Dict[int, Branch]]:
        self._surface_points = surface_points
        self._skeleton = skeleton
        self._skeletal_points = skeletal_points
        self._radii = radii
        self._z_min = np.min(self._surface_points[:, 2])
        self._tree_height = np.max(self._surface_points[:, 2]) - self._z_min
        self._surface_points = None

        for fn in self._pipeline:
            yield fn()
        
        self._clear()
        
        return self._branch_id_to_branch