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
from utils.networkx_extra import WeightedPathLength, GreedyPathPartitioning
import open3d as o3d
from sklearn.isotonic import IsotonicRegression
from .data_type import Branch
from scipy.spatial import KDTree
from utils.fem_pos_deviation_osqp_interface import FemPosDeviationOsqpInterface
from utils.get_distinct_colors import get_distinct_colors
from utils.open3d_extra import create_cylinder
from scipy.optimize import Bounds
from utils.scipy_extra import wrapped_minimize
from .pipeline import Pipeline
from utils.scipy_extra import MonotoneISplineQP

class Refinement(Pipeline):
    _trusted_aboveground_height: Optional[float]
    _trusted_height_ratio: Optional[float]
    _min_radius_weight: Optional[float]
    _x0_on_gamma_dist: Optional[float]
    _y0_on_gamma_dist: Optional[float]
    _wpl_based_correction_fn_for_radius_weights: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    _min_radius: Optional[float]

    _occupancy_factor_or_buffer: Optional[float]

    _smoother_kwargs: Optional[Dict[str, Any]]

    _radius_tolerance: Optional[float]
    _using_allometric_equation: Optional[bool]
    _allometric_eq_without_intercept: Callable[[Union[np.ndarray, float], List[float]], Union[np.ndarray, float]]
    _allometric_parameter_initvals: Optional[List[float]]
    _allometric_parameter_minvals: Optional[List[float]]
    _allometric_parameter_maxvals: Optional[List[float]]
    _trunk_intercept_initval: Optional[float]
    _trunk_intercept_minval: Optional[float]
    _trunk_intercept_maxval : Optional[float]
    _lam_intercept_minval: Optional[float]
    _lam_intercept_maxval: Optional[float]
    _lam_intercept_tuning_fn: Optional[Callable[[float, float, float, float], float]]
    _min_num_credible_samples: Optional[int]
    _making_twigs_thinner: Optional[bool]

    _z_min: Optional[float]
    _tree_height: Optional[float]
    _surface_points: Optional[np.ndarray]
    _skeleton: Optional[nx.DiGraph]
    _skeletal_points: Optional[np.ndarray]
    _radii: Optional[np.ndarray]

    _wpls: Optional[np.ndarray] # Weighted path lengths or growth lengths (of node i) or total branch lengths (supported by node i)
    _paths: Optional[List[List[int]]]
    _reference_radii: Optional[np.ndarray]
    _path_idx_to_branch: Optional[Dict[int, Branch]]
    _branch_id_to_branch: Optional[Dict[int, Branch]]

    def _clear(self) -> None:
        self._trusted_aboveground_height = None
        self._trusted_height_ratio = None
        self._min_radius_weight = None
        self._x0_on_gamma_pdf = None
        self._y0_on_gamma_pdf = None
        self._wpl_based_correction_fn_for_radius_weights = None
        self._min_radius = None

        self._occupancy_factor_or_buffer = None

        self._smoother_kwargs = None

        self._radius_tolerance = None
        self._using_allometric_equation = None
        self._allometric_eq_without_intercept = None
        self._allometric_parameter_initvals = None
        self._allometric_parameter_minvals = None
        self._allometric_parameter_maxvals = None
        self._trunk_intercept_minval = None
        self._trunk_intercept_maxval  = None
        self._trunk_intercept_initval = None
        self._lam_intercept_minval = None
        self._lam_intercept_maxval = None
        self._lam_intercept_tuning_fn = None
        self._min_num_credible_samples = None
        self._making_twigs_thinner = None

        self._z_min = None
        self._tree_height = None
        self._surface_points = None
        self._skeleton = None
        self._skeletal_points = None
        self._radii = None
        
        self._wpls = None
        self._paths = None
        self._reference_radii = None
        self._path_idx_to_branch = None
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
            wpl_based_correction_fn_for_radius_weights: Union[Callable[[np.ndarray, np.ndarray], np.ndarray], str] = lambda radius_weights, wpls: radius_weights * (np.log(wpls + 1.0) / np.log(np.max(wpls) + 1.0)),
            min_radius: float = 1.0e-3,
            occupancy_factor_or_buffer: float = 1.1,
            smoother_kwargs: Optional[Dict[str, Any]] = {},
            radius_tolerance: float = 0.1,
            using_allometric_equation: bool = True,
            allometric_equation_without_intercept: Union[Callable[[Union[np.ndarray, float], List[float]], Union[np.ndarray, float]], str] = lambda x, c: c[0] * x ** c[1],
            allometric_parameter_initvals: Optional[List[float]] = [0.01, 0.3],
            allometric_parameter_minvals: Optional[List[float]] = [1.0e-10, 0.1],
            allometric_parameter_maxvals: Optional[List[float]] = [1.0e10, 0.5],
            trunk_intercept_minval: float = 0.0,
            trunk_intercept_maxval : float = 1.0e10,
            trunk_intercept_initval: float = 0.0,
            lam_intercept_minval: float = 0.1,
            lam_intercept_maxval: float = 100.0,
            lam_intercept_tuning_function: Optional[Callable[[float, float, float, float], float]] = lambda wpl, wpl_max, lam_lb, lam_ub: (lam_lb / lam_ub) ** (np.log(wpl + 1.0) / np.log(wpl_max + 1.0)) * lam_ub,
            min_num_credible_samples: int = 5,
            making_twigs_thinner: bool = True
    ) -> None:
        if min_radius <= 0.:
            raise ValueError("min_radius must be > 0.")
        self._min_radius = min_radius
        if radius_tolerance < 0. or radius_tolerance > 1.:
            raise ValueError("radius_tolerance must be >= 0. and <= 1.")
        self._radius_tolerance = radius_tolerance
        if occupancy_factor_or_buffer <= 0.:
            raise ValueError("occupancy_factor_or_buffer must be > 0, where [0, 1) is buffer size and >=1 is multiplier.")
        self._occupancy_factor_or_buffer = occupancy_factor_or_buffer
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
        
        self._smoother_kwargs = smoother_kwargs

        self._using_allometric_equation = using_allometric_equation
        if callable(allometric_equation_without_intercept):
            self._allometric_eq_without_intercept = allometric_equation_without_intercept
        else:
            self._allometric_eq_without_intercept = eval(allometric_equation_without_intercept)
        self._allometric_parameter_initvals = allometric_parameter_initvals
        self._allometric_parameter_minvals = allometric_parameter_minvals
        self._allometric_parameter_maxvals = allometric_parameter_maxvals
        self._trunk_intercept_minval = trunk_intercept_minval
        self._trunk_intercept_maxval  = trunk_intercept_maxval 
        self._trunk_intercept_initval = trunk_intercept_initval

        self._lam_intercept_minval = lam_intercept_minval
        self._lam_intercept_maxval = lam_intercept_maxval
        if callable(lam_intercept_tuning_function):
            self._lam_intercept_tuning_fn = lam_intercept_tuning_function
        else:
            self._lam_intercept_tuning_fn = eval(lam_intercept_tuning_function)

        if min_num_credible_samples <= 0:
            raise ValueError("min_num_credible_samples must be > 0.")
        self._min_num_credible_samples = min_num_credible_samples
        self._making_twigs_thinner = making_twigs_thinner

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
            self._update_skeleton,
            self._estimate_radii,
        ])
        if not self._using_allometric_equation:
            super()._add_fns_to_pipeline(len(self._pipeline), [
                self._regenerate_branches_by_thickness,
            ])
        super()._add_fns_to_pipeline(len(self._pipeline), [
            self._smooth_pathwise,
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
        for path_idx, path in enumerate(self._paths):
            for i in range(len(path) - 1):
                edges.append((path[i], path[i + 1]))
                edge_colors.append(path_colors[path_idx])
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(edge_colors)
        return f"Partitioned the skeleton into {len(self._paths)} paths.", lineset

    def _generate_branches(self) -> Optional[Tuple[str, o3d.geometry.LineSet]]:
        branch_orders_of_node: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        path_indices_of_node: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        self._path_idx_to_branch = {}
        for path_idx, path in enumerate(self._paths):
            branch: Branch = Branch()
            parent_branch_id: int = -1
            branch_order: int = 0
            joint_point_idx: int = -1
            if path_idx != 0:
                parent_branch_id = path_indices_of_node[path[0]]
                branch_order = branch_orders_of_node[path[0]] + 1
                joint_point_idx = path[0]
                path = path[1:] # Remove the joint point from each branch
            branch.order = branch_order
            branch.parent_id = parent_branch_id
            branch.base_radius = self._skeletal_points[path[0], 2] # Temporarily borrow this attribute to store height for sorting
            branch.joint_point_idx = joint_point_idx
            self._path_idx_to_branch[path_idx] = branch
            branch_orders_of_node[path] = branch_order
            path_indices_of_node[path] = path_idx
            self._paths[path_idx] = path

        # Sort dictionary in ascending sequence of order and descending sequence of height
        self._path_idx_to_branch = dict(sorted(
            self._path_idx_to_branch.items(), 
            key=lambda item: (item[1].order, -item[1].base_radius)
        ))

        if not self._verbose:
            return
        
        max_branch_order: int = 0
        for path_idx, branch in self._path_idx_to_branch.items():
            max_branch_order = max(max_branch_order, branch.order)

        branch_order_colors: List[Tuple[float, float, float]] = get_distinct_colors(max_branch_order + 1)
        
        edges: List[Tuple[int, int]] = []
        edge_colors: List[Tuple[float, float, float]] = []
        for path_idx, branch in self._path_idx_to_branch.items():
            path: List[int] = self._paths[path_idx]
            branch_order: int = branch.order
            for j in range(len(path) - 1):
                edges.append((path[j], path[j + 1]))
                edge_colors.append(branch_order_colors[branch_order])

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(edge_colors)

        return f"Generated {len(self._path_idx_to_branch)} branches (Max branch order={max_branch_order}).", lineset
    
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
        
        weights[self._radii == 0.0] = self._min_radius_weight # Compared to post mask fitting, suppressing weights is more robust in extreme cases

        self._reference_radii = np.zeros(len(self._skeletal_points))

        for path_idx, branch in self._path_idx_to_branch.items():
            path: List[int] = self._paths[path_idx]
            if path_idx != 0:
                path.insert(0, branch.joint_point_idx)
            pathwise_lengths: np.ndarray = self._wpls[path]
            pathwise_radii: np.ndarray = self._radii[path].copy()
            pathwise_weights: np.ndarray = weights[path].copy()
            
            if path_idx != 0:
                pathwise_radii[0] = self._reference_radii[path[0]]
                pathwise_weights[0] = 1e4
            
            pathwise_reference_radii: np.ndarray = IsotonicRegression().fit_transform(
                pathwise_lengths, 
                pathwise_radii, 
                sample_weight = pathwise_weights
            )
            if path_idx != 0:
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
        for point_idx, point in enumerate(self._skeletal_points):
            radius: float = self._reference_radii[point_idx]
            sphere_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
            sphere_mesh.translate(point)
            sphere_mesh.compute_vertex_normals()
            mesh += sphere_mesh
        
        return f"Calculated the reference radii (max={np.max(self._reference_radii):.4f}).", mesh

    def _identify_reasonable_branches(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        # Thicker branches in the same order are given priority
        for path_idx, branch in self._path_idx_to_branch.items():
            branch.base_radius = self._reference_radii[self._paths[path_idx][0]]
        self._path_idx_to_branch = dict(sorted(
            self._path_idx_to_branch.items(), 
            key=lambda item: (item[1].order, -item[1].base_radius)
        ))
        
        # Occupy nodes with a series of balls along each path
        path_indices_of_node = -np.ones(len(self._skeletal_points), dtype=int)
        after_check: np.ndarray = np.zeros(len(self._skeletal_points), dtype=bool)
        kdtree: KDTree = KDTree(self._skeletal_points)
        for path_idx in list(self._path_idx_to_branch.keys()):
            path: List[int] = self._paths[path_idx]
            parent_branch_id: int = -1
            joint_node: int = -1
            if path_idx > 0:
                # Remove nodes at the furcation
                while len(path) > 0:
                    if after_check[path[0]] == True: 
                        path = path[1:]
                    else:
                        break
                
                # Remove all intermediate occupied nodes
                # The remaining part maintains continuity
                checked_id: int = 0
                while checked_id < len(path):
                    if after_check[path[checked_id]] == False:
                        checked_id += 1
                    else:
                        path = path[:checked_id] + path[checked_id + 1:]
                
            self._paths[path_idx] = path

            if path_idx > 0:
                if len(path) <= 1:
                    del self._path_idx_to_branch[path_idx]
                    continue

                # Find parent branch
                joint_node = list(self._skeleton.predecessors(path[0]))[0]
                while True:
                    if path_indices_of_node[joint_node] != -1 and after_check[joint_node] == True:
                        parent_branch_id = path_indices_of_node[joint_node]
                        break
                    predecessors: List[int] = list(self._skeleton.predecessors(joint_node))
                    joint_node = predecessors[0] # It will definitely not be empty
                    
            self._path_idx_to_branch[path_idx].parent_id = parent_branch_id
            self._path_idx_to_branch[path_idx].joint_point_idx = joint_node # For temporary use
            if joint_node != -1:
                self._path_idx_to_branch[path_idx].order = self._path_idx_to_branch[parent_branch_id].order + 1
            
            neighbor_ids_per_node: List[List[int]] = kdtree.query_ball_point(
                self._skeletal_points[path], 
                r=self._reference_radii[path] * self._occupancy_factor_or_buffer if self._occupancy_factor_or_buffer >= 1. else self._reference_radii[path] + self._occupancy_factor_or_buffer
            )

            inlier_nodes: np.ndarray = np.unique(
                np.concatenate(neighbor_ids_per_node)
            )
            
            after_check[inlier_nodes] = True
        
            path_indices_of_node[path] = path_idx

        if not self._verbose:
            return

        path_colors: List[Tuple[float, float, float]] = get_distinct_colors(len(self._path_idx_to_branch))
        
        edges: List[Tuple[int, int]] = []
        edge_colors: List[Tuple[float, float, float]] = []
        for i, path_idx in enumerate(self._path_idx_to_branch.keys()):
            path: List[int] = self._paths[path_idx]
            for j in range(len(path) - 1):
                edges.append((path[j], path[j + 1]))
                edge_colors.append(path_colors[i])

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(edge_colors)

        return f"Identified {len(self._path_idx_to_branch)} reasonable branches.", lineset
    
    def _update_skeleton(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.LineSet]]:
        self._skeleton = nx.DiGraph()
        for path_idx, branch in self._path_idx_to_branch.items():
            path: List[int] = self._paths[path_idx]
            for i in range(len(path) - 1):
                self._skeleton.add_edge(path[i], path[i + 1], weight=np.linalg.norm(self._skeletal_points[path[i]] - self._skeletal_points[path[i + 1]]))
            if branch.parent_id != -1:
                self._skeleton.add_edge(branch.joint_point_idx, path[0], weight=np.linalg.norm(self._skeletal_points[branch.joint_point_idx] - self._skeletal_points[path[0]]))

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

        weight_fem_pos_deviation: float = self._smoother_kwargs.get("weight_fem_pos_deviation", 1.0e10)
        weight_path_length: float = self._smoother_kwargs.get("weight_path_length", 1.0)
        weight_ref_deviation: float = self._smoother_kwargs.get("weight_ref_deviation", 1.0)
        bound: float = self._smoother_kwargs.get("bound", 0.2)

        
        fem_pos_deviation_smoother: FemPosDeviationOsqpInterface = FemPosDeviationOsqpInterface()
        fem_pos_deviation_smoother.set_weight_fem_pos_deviation(weight_fem_pos_deviation)
        fem_pos_deviation_smoother.set_weight_path_length(weight_path_length)
        fem_pos_deviation_smoother.set_weight_ref_deviation(weight_ref_deviation)
        
        count: int = 0
        for path_idx in self._path_idx_to_branch.keys():
            path: List[int] = self._paths[path_idx]
            if len(path) < 3:
                continue
            elif len(path) == 3:
                updated_skeletal_points[path[1]] = (self._skeletal_points[path[0]] + self._skeletal_points[path[2]]) / 2.0
                continue
            count += 1
            
            points: np.ndarray = self._skeletal_points[path]
            
            bounds: np.ndarray = np.full(len(points), bound)
            
            bounds[0] = 0.0
            bounds[-1] = 0.0

            fem_pos_deviation_smoother.set_ref_points(points)
            fem_pos_deviation_smoother.set_bounds_around_refs(bounds)
            if fem_pos_deviation_smoother.Solve():
                updated_skeletal_points[path] = fem_pos_deviation_smoother.opt_points_

        # Smoothed branching joints have shifted
        for path_idx, branch in self._path_idx_to_branch.items():
            parent_branch_id: int = branch.parent_id
            if parent_branch_id == -1:
                continue
            path: List[int] = self._paths[path_idx]
            parent_path: List[int] = self._paths[parent_branch_id]
            kdtree: KDTree = KDTree(updated_skeletal_points[parent_path])
            joint_node: int = parent_path[kdtree.query(self._skeletal_points[branch.joint_point_idx], k=1)[1]]
            self._path_idx_to_branch[path_idx].joint_point_idx = joint_node
        
        self._skeletal_points = updated_skeletal_points

        if not self._verbose:
            return
        
        edges: List[Tuple[int, int]] = []
        for path_idx in self._path_idx_to_branch.keys():
            for i in range(len(self._paths[path_idx]) - 1):
                edges.append(
                    (self._paths[path_idx][i], self._paths[path_idx][i + 1])
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
        path_idx_to_intercept: Dict[int, float] = {}

        if self._using_allometric_equation:
            # Allometric equation:
            # y=a*(ln(1+x))^b X=ln(1+x):         b0=1.1 in [0.5,  1.5 ] a0=0.01 O(log(x))
            # y=a*(x/(ln(x+e)))^b X=x/(ln(x+e)): b0=0.4 in [0.15, 1   ] a0=0.01 O(x/log(x))
            # y=a*x^b X=x:                       b0=0.3 in [0.1,  0.6 ] a0=0.01 O(x)
            weights[
                (self._radii < self._reference_radii * (1 - self._radius_tolerance))
                | (self._radii > self._reference_radii * (1 + self._radius_tolerance))
            ] = 0.0

            allometric_parameters: np.ndarray

            for path_idx, branch in self._path_idx_to_branch.items():
                path: List[int] = self._paths[path_idx]

                lam_intercept: float = self._lam_intercept_tuning_fn(
                    self._wpls[path[0]], 
                    stem_length, 
                    self._lam_intercept_minval, 
                    self._lam_intercept_maxval
                )

                if path_idx == 0:
                    sample_weights: np.ndarray = weights[path]

                    # Fit y=ax^b+c
                    # Optimize \min \sum_i^n w_i (a x_i ^ b + c - y_i)^2 / \sum_i^n w_i + lambda * c^2
                    c: np.ndarray = wrapped_minimize(
                        lambda c, x, y, w, lambda_squared_intercept: \
                            np.sum(
                                w * (y - (self._allometric_eq_without_intercept(x, c[:-1]) + c[-1]))**2
                            ) / np.sum(w) \
                            + lambda_squared_intercept * c[2] ** 2,
                        x0=self._allometric_parameter_initvals + [self._trunk_intercept_initval],
                        args=(self._wpls[path], self._radii[path], sample_weights, lam_intercept),
                        bounds=Bounds(
                            lb=self._allometric_parameter_minvals + [self._trunk_intercept_minval],
                            ub=self._allometric_parameter_maxvals + [self._trunk_intercept_maxval ]
                        ),
                        method="L-BFGS-B",
                        options={
                            "maxiter": 5000,
                            "ftol": 1e-12
                        }
                    )

                    self._radii[path] = np.clip(
                        self._allometric_eq_without_intercept(self._wpls[path], c[:-1]) + c[-1], 
                        self._min_radius, 
                        np.inf
                    )
                    allometric_parameters = c[:-1]
                    path_idx_to_intercept[path_idx] = c[-1]
                    self._path_idx_to_branch[path_idx].base_radius = np.nan
                else:
                    # Given a0, b0 and c0, fit y=a0x^b0+c where initial c=c0 and c is restricted by the parent branch: 
                    # The thickness of the sub branch must not exceed the subsequent thickness of the parent branch at the branching joint
                    sample_weights: np.ndarray = weights[path]

                    joint_node: int = branch.joint_point_idx
                    parent_branch_id: int = branch.parent_id
                    
                    intercept: float = path_idx_to_intercept[parent_branch_id]
                    max_growth_length: float = 0.
                    growth_length: float = 0.
                    for successor in list(self._skeleton.successors(joint_node)):
                        wpl: float = self._wpls[successor] + np.linalg.norm(self._skeletal_points[successor] - self._skeletal_points[joint_node])
                        if successor == path[0]:
                            growth_length = wpl
                        max_growth_length = max(
                            max_growth_length,
                            wpl
                        )

                    if np.sum(sample_weights != 0.0) >= self._min_num_credible_samples:
                        max_intercept = intercept + self._allometric_eq_without_intercept(
                            max_growth_length, 
                            allometric_parameters
                        ) - self._allometric_eq_without_intercept(
                            growth_length, 
                            allometric_parameters
                        )

                        c: np.ndarray = wrapped_minimize(
                            fun=lambda c, x, y, w, lambda_squared_intercept, allometric_parameters=allometric_parameters: \
                                np.sum(
                                    w * (y - (self._allometric_eq_without_intercept(x, allometric_parameters) + c[0])) ** 2 
                                ) / np.sum(w) \
                                + lambda_squared_intercept * c[0] ** 2,
                            x0=[intercept],
                            bounds=Bounds(lb=[-np.inf], ub=[max_intercept]),
                            args=(
                                self._wpls[path],
                                self._radii[path],
                                sample_weights,
                                lam_intercept
                            ),
                            options={
                                "maxiter": 5000,
                                "ftol": 1e-12
                            }
                        )

                        intercept = c[0]
                    else:
                        if self._making_twigs_thinner:
                            intercept = min(intercept, path_idx_to_intercept[0])

                    path_idx_to_intercept[path_idx] = intercept
                    self._radii[path] = np.clip(self._allometric_eq_without_intercept(self._wpls[path], allometric_parameters) + intercept, self._min_radius, np.inf)
                    self._path_idx_to_branch[path_idx].base_radius = max(
                        self._allometric_eq_without_intercept(growth_length, allometric_parameters) + intercept, 
                        self._min_radius
                    )
        else:
            # Smooth monotonic regression results using spline curves
            path_idx_to_spline: Dict[int, MonotoneISplineQP] = {}
            radii: np.ndarray = self._reference_radii.copy()
            mask: np.ndarray = (self._radii >= self._reference_radii * (1 - self._radius_tolerance)) & (self._radii <= self._reference_radii * (1 + self._radius_tolerance))
            radii[mask] = self._radii[mask]

            for path_idx, branch in self._path_idx_to_branch.items():
                path: List[int] = self._paths[path_idx]

                lam_intercept: float = self._lam_intercept_tuning_fn(
                    self._wpls[path[0]], 
                    stem_length, 
                    self._lam_intercept_minval, 
                    self._lam_intercept_maxval
                )

                pathwise_wpls: np.ndarray = self._wpls[path][::-1] # small to large
                pathwise_radii: np.ndarray = radii[path][::-1]
                pathwise_weights: np.ndarray = weights[path][::-1]
                
                num_steps: int = np.unique(pathwise_radii).size
                
                if path_idx == 0:
                    pathwise_weights /= np.sum(pathwise_weights)
                    pathwise_wpls = np.insert(pathwise_wpls, 0, 0.0)
                    pathwise_radii = np.insert(pathwise_radii, 0, self._trunk_intercept_initval)
                    pathwise_weights = np.insert(pathwise_weights, 0, lam_intercept)

                    spline: MonotoneISplineQP = MonotoneISplineQP(pathwise_wpls, pathwise_radii,  k=min(3, num_steps - 1), w=pathwise_weights, n_basis=num_steps)
                    
                    self._radii[path] = np.clip(spline(pathwise_wpls[1:]), self._min_radius, None)[::-1] # large to small
                    path_idx_to_intercept[path_idx] = spline(0.0)
                    self._path_idx_to_branch[path_idx].base_radius = np.nan
                    path_idx_to_spline[path_idx] = spline
                else:
                    growth_length: float = self._wpls[path[0]] + np.linalg.norm(self._skeletal_points[branch.joint_point_idx] - self._skeletal_points[path[0]])
                    
                    pathwise_wpls = np.append(pathwise_wpls, self._wpls[branch.joint_point_idx])
                    pathwise_radii = np.append(pathwise_radii, self._radii[branch.joint_point_idx])
                    pathwise_weights = np.append(pathwise_weights, weights[branch.joint_point_idx])
                    pathwise_weights /= np.sum(pathwise_weights)

                    pathwise_wpls = np.insert(pathwise_wpls, 0, 0.0)
                    pathwise_radii = np.insert(pathwise_radii, 0, path_idx_to_intercept[branch.parent_id])
                    pathwise_weights = np.insert(pathwise_weights, 0, lam_intercept)

                    spline: MonotoneISplineQP = MonotoneISplineQP(pathwise_wpls, pathwise_radii,  k=min(3, num_steps - 1), w=pathwise_weights, n_basis=num_steps)
                    
                    self._radii[path] = np.clip(spline(pathwise_wpls[1:-1]), self._min_radius, self._radii[branch.joint_point_idx])[::-1] # large to small

                    path_idx_to_intercept[path_idx] = spline(0.0)
                    self._path_idx_to_branch[path_idx].base_radius = max(
                        self._min_radius, 
                        min(
                            max(
                                spline(growth_length), 
                                spline(pathwise_wpls[-2])
                            ), 
                            self._radii[branch.joint_point_idx]
                        )
                    )
                    path_idx_to_spline[path_idx] = spline
                    

        for path_idx, branch in self._path_idx_to_branch.items():
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

        for path_idx, branch in self._path_idx_to_branch.items():
            path: List[int] = self._paths[path_idx]
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

    def _regenerate_branches_by_thickness(self):
        node_to_radius: Dict[int, float] = {}
        for i, radius in enumerate(self._radii):
            node_to_radius[i] = radius

        paths = GreedyPathPartitioning(
            self._skeleton, 
            node_to_radius, 
            maximized=True
        ).get_paths()

        branch_orders_of_node: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        path_indices_of_node: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        self._path_idx_to_branch = {}
        for path_idx, path in enumerate(paths):
            branch: Branch = Branch()
            parent_branch_id: int = -1
            branch_order: int = 0
            joint_point_idx: int = -1
            if path_idx != 0:
                parent_branch_id = path_indices_of_node[path[0]]
                branch_order = branch_orders_of_node[path[0]] + 1
                joint_point_idx = path[0]
                path = path[1:] # Remove the joint point from each branch
            branch.order = branch_order
            branch.parent_id = parent_branch_id
            branch.base_radius = self._radii[path[0]] 
            branch.joint_point_idx = joint_point_idx
            self._path_idx_to_branch[path_idx] = branch
            branch_orders_of_node[path] = branch_order
            path_indices_of_node[path] = path_idx
            self._paths[path_idx] = path

        self._path_idx_to_branch = dict(sorted(
            self._path_idx_to_branch.items(), 
            key=lambda item: (item[1].order, -item[1].base_radius)
        ))

        if not self._verbose:
            return
        
        max_branch_order: int = 0
        for path_idx, branch in self._path_idx_to_branch.items():
            max_branch_order = max(max_branch_order, branch.order)

        branch_order_colors: List[Tuple[float, float, float]] = get_distinct_colors(max_branch_order + 1)
        
        edges: List[Tuple[int, int]] = []
        edge_colors: List[Tuple[float, float, float]] = []
        for path_idx, branch in self._path_idx_to_branch.items():
            path: List[int] = self._paths[path_idx]
            branch_order: int = branch.order
            for j in range(len(path) - 1):
                edges.append((path[j], path[j + 1]))
                edge_colors.append(branch_order_colors[branch_order])

        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._skeletal_points)
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(edge_colors)

        return f"Generated {len(self._path_idx_to_branch)} branches (Max branch order={max_branch_order}).", lineset

    def _convert_data_format(self) -> Optional[Tuple[str, o3d.geometry.TriangleMesh]]:
        # For safety reasons, check if the adjacent points of each branch are too close
        node_to_new_node: Dict[int, Dict[int, int]] = {}
        path_idx_to_new_path_idx_with_joint_id: Dict[int, Tuple[int, int]] = {}

        for path_idx in list(self._path_idx_to_branch.keys()):
            branch: Branch = self._path_idx_to_branch[path_idx]

            if branch.parent_id in path_idx_to_new_path_idx_with_joint_id:
                branch.parent_id, branch.joint_point_idx = path_idx_to_new_path_idx_with_joint_id[branch.parent_id]
                branch.order = self._path_idx_to_branch[branch.parent_id].order + 1
            
            if branch.joint_point_idx in node_to_new_node:
                branch.joint_point_idx = node_to_new_node[branch.joint_point_idx]

            path: List[int] = self._paths[path_idx]
            i: int = len(path) - 2
            while i >= 0:
                if np.linalg.norm(
                    self._skeletal_points[path[i]] - self._skeletal_points[path[i + 1]]
                ) < 1e-3:
                    node_to_new_node[path[i + 1]] = path[i]
                    del path[i + 1]
                i -= 1
                    
            if len(path) < 2:
                path_idx_to_new_path_idx_with_joint_id[path_idx] = (
                    branch.parent_id,
                    branch.joint_point_idx
                )
                del self._path_idx_to_branch[path_idx]
                
        # No longer using index structures in Branch data type
        ids_on_each_path: np.ndarray = -np.ones(len(self._skeletal_points), dtype=int)
        
        branch_id_to_branch: Dict[int, Branch] = {}
        for path_idx, branch in self._path_idx_to_branch.items():
            path: List[int] = self._paths[path_idx]
            new_branch: Branch = Branch()
            new_branch.medial_points = self._skeletal_points[path]
            new_branch.radii = self._radii[path]
            new_branch.order = branch.order
            new_branch.base_radius = branch.base_radius # radius of branching joint
            if branch.parent_id != -1:
                new_branch.joint_point_idx = ids_on_each_path[branch.joint_point_idx]
                new_branch.parent_id = branch.parent_id + 1 # Match TreeQSM
            else:
                new_branch.joint_point_idx = -1
                new_branch.parent_id = -1
            branch_id_to_branch[path_idx + 1] = new_branch # Match TreeQSM
            ids_on_each_path[path] = np.linspace(0, len(path) - 1, len(path), dtype=int)

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