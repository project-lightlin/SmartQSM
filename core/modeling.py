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

from .data_type import Branch
from .pipeline import Pipeline
from utils.arterial_snake import generate_arterial_snake
from utils.get_distinct_colors import get_distinct_colors
from typing import Dict, Generator, Any, List, Tuple, Optional
import open3d as o3d
import numpy as np
from utils.numpy_extra import calculate_distances_from_points_to_segments_3d, calculate_symmetric_points, normalize
from utils.rdp import rdp_fast
from utils.hermite_curve import hermite_curve
from scipy.spatial import KDTree
import math

class Modeling(Pipeline):
    _branch_id_to_branch: Optional[Dict[int, Branch]]
    _num_sectional_vertices: Optional[int] 
    _min_radius: Optional[float]
    _resolution: Optional[float]
    _num_sectional_vertices_for_optimization: Optional[int]
    _rdp_epsilon: Optional[float]
    _num_hermite_nodes: Optional[int]

    def _clear(self) -> None:
        self._num_sectional_vertices = None
        self._min_radius = None
        self._resolution = None
        self._range_multiplier = None
        self._num_sectional_vertices_for_optimization = None
        self._rdp_epsilon = None
        self._num_hermite_nodes = None
        return

    def __init__(self, *, verbose: bool = False) -> None:
        self._branch_id_to_branch = None

        self._clear()

        super().__init__(
            verbose=verbose
        )
        return
    
    def set_params(self, *, num_sectional_vertices: int = 20,  using_furcation_optimization: bool = True, min_radius: float = 0.001, resolution: float = 0.01, range_multiplier: float = 2.0, num_sectional_vertices_for_optimization: int = 12, rdp_epsilon: float = 0.01, num_hermite_nodes: int = 8) -> None:
        if num_sectional_vertices < 3 or num_sectional_vertices_for_optimization < 3:
            raise ValueError("num_sectional_vertices must be >= 3")
        self._num_sectional_vertices = num_sectional_vertices
        self._min_radius = min_radius
        self._resolution = resolution
        self._range_multiplier = range_multiplier
        self._num_sectional_vertices_for_optimization = num_sectional_vertices_for_optimization
        self._rdp_epsilon = rdp_epsilon
        self._num_hermite_nodes = num_hermite_nodes

        self._clear_pipeline()
        self._add_fns_to_pipeline(len(self._pipeline), [
            self._optimize_furcation if using_furcation_optimization else self._repair_furcations,
            self._model
        ])
        return
    
    def _repair_furcations(self) -> Optional[Tuple[str, o3d.geometry.LineSet]]:
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch.parent_id == -1:
                continue
            joint_point_idx: int = branch.joint_point_idx
            branch.medial_points = np.vstack((
                self._branch_id_to_branch[branch.parent_id].medial_points[joint_point_idx], 
                branch.medial_points
            ))
            branch.radii = np.concatenate(([branch.base_radius], branch.radii))

        self._interpolate_joint_points()
        
        if not self._verbose:
            return
        
        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        points: List[np.ndarray] = []
        lines: List[Tuple[int, int]] = []
        colors: List[np.ndarray] = []

        for branch_id, branch in self._branch_id_to_branch.items():
            base_point_idx: int = len(points)
            points.extend(branch.medial_points)
            num_points: int = len(branch.medial_points)
            lines.extend(zip(
                np.arange(num_points - 1) + base_point_idx,
                np.arange(num_points - 1) + 1 + base_point_idx
            ))
            branch_colors: np.ndarray = np.zeros((num_points - 1, 3), dtype=float)
            if branch.parent_id != -1:
                branch_colors[0, :] = np.array([1., 0., 0.])
            colors.extend(branch_colors)
        
        lineset.points = o3d.utility.Vector3dVector(np.array(points))
        lineset.lines = o3d.utility.Vector2iVector(np.array(lines))
        lineset.colors = o3d.utility.Vector3dVector(np.array(colors))

        return f"Repaired furcations.", lineset


    def _optimize_furcation(self) -> Optional[Tuple[str, o3d.geometry.LineSet]]:
        optimized_branch_ids: Optional[List[float]] = []
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch.parent_id == -1:
                continue
            parent_branch: Branch = self._branch_id_to_branch[branch.parent_id]
            
            direction: np.ndarray = normalize(branch.medial_points[1] - branch.medial_points[0])
            
            # Search area should be no less than the sphere from the closest point on the parent branch skeleton to the beginning of the current branch
            segment_starts: np.ndarray = parent_branch.medial_points[:-1]
            segment_ends: np.ndarray = parent_branch.medial_points[1:]
            closest_distances_per_point: np.ndarray
            closest_points_per_point: np.ndarray
            closest_distances_per_point, closest_points_per_point, _ = calculate_distances_from_points_to_segments_3d(
                [branch.medial_points[0]],
                segment_starts,
                segment_ends
            )
            point_idx_with_closest_distance: int = np.argmin(closest_distances_per_point[0, :])
            sphere_center: np.ndarray = closest_points_per_point[0, point_idx_with_closest_distance]
            sphere_radius: float = np.linalg.norm(sphere_center - branch.medial_points[0])
            kdtree: KDTree = KDTree(parent_branch.medial_points)
            neighbor_parent_skeletal_nodes: np.ndarray = np.array(kdtree.query_ball_point(
                sphere_center, 
                sphere_radius * self._range_multiplier
            ))
            available_parent_sketetal_nodes: np.ndarray
            if len(neighbor_parent_skeletal_nodes) != 0:
                available_parent_sketetal_nodes = np.arange(
                    max(np.min(neighbor_parent_skeletal_nodes) - 1, 0), 
                    min(np.max(neighbor_parent_skeletal_nodes) + 2, len(parent_branch.medial_points))
                )
            else:
                available_parent_sketetal_nodes = np.arange(0, len(parent_branch.medial_points))

            # Generate possible surfaces for connection points
            # The feasible surface is defined by a slimmer “snake” whose radius equals the cross-sectional radius minus the base radius, preventing clipping between each branch base and its parent branch.
            mesh: o3d.geometry.TriangleMesh = generate_arterial_snake(
                parent_branch.medial_points[available_parent_sketetal_nodes], 
                np.clip(parent_branch.radii[available_parent_sketetal_nodes] - branch.base_radius, a_min=self._min_radius, a_max=None), 
                self._num_sectional_vertices_for_optimization
            )
            
            # Directly maintaining the angle is the most stable way
            mesh_area: float = mesh.get_surface_area()
            if math.isnan(mesh_area):
                continue
            
            num_points: int = int(np.ceil(mesh_area * (1 / self._resolution) * (1 / self._resolution) * (4 / 3.14)))
            cloud: o3d.geometry.PointCloud = mesh.sample_points_uniformly(
                number_of_points=num_points,
                use_triangle_normal=True
            )
            
            # Pick out points (on M) within spherical boundary
            points: np.ndarray = np.asarray(cloud.points)
            normals: np.ndarray = np.asarray(cloud.normals)
            kdtree = KDTree(points)
            neighbor_point_idx: np.ndarray = np.array(kdtree.query_ball_point(
                sphere_center,
                sphere_radius
            ))

            if neighbor_point_idx.shape[0] == 0:
                # Strange case
                branch.medial_points = np.vstack((
                    self._branch_id_to_branch[branch.parent_id].medial_points[branch.joint_point_idx], 
                    branch.medial_points
                ))
                branch.radii = np.concatenate(([branch.base_radius], branch.radii))
                continue

            points = points[neighbor_point_idx]
            normals = normals[neighbor_point_idx]

            # Filter out the points on the back
            # Due to the potential connection to further points, it is important to set a suitable sphere radius coefficient to reduce risk
            mask = np.dot(normals, direction) <= 0
            points = points[mask]

            if points.shape[0] == 0:
                # Strange case
                branch.medial_points = np.vstack((
                    self._branch_id_to_branch[branch.parent_id].medial_points[branch.joint_point_idx], 
                    branch.medial_points
                ))
                branch.radii = np.concatenate(([branch.base_radius], branch.radii))
                continue
            
            costs: np.ndarray = 1. - np.dot(normalize(branch.medial_points[0] - points), direction)
                
            id: int = np.argmin(costs)
            best_connection_point: np.ndarray = points[id]

            branch.medial_points = np.vstack((best_connection_point, branch.medial_points))
            branch.radii = np.concatenate(([branch.base_radius], branch.radii))

            kdtree = KDTree(parent_branch.medial_points)
            branch.joint_point_idx = kdtree.query(best_connection_point, k=1)[1]

            optimized_branch_ids.append(branch_id)

        self._interpolate_joint_points()

        # Radius on branch must be less than the radius at the new furcation
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch.parent_id == -1:
                continue
            parent_branch: Branch = self._branch_id_to_branch[branch.parent_id]
            branch.radii = np.clip(branch.radii, None, parent_branch.radii[branch.joint_point_idx])

        if not self._verbose:
            return
        
        lineset: o3d.geometry.LineSet = o3d.geometry.LineSet()
        points: List[np.ndarray] = []
        lines: List[Tuple[int, int]] = []
        colors: List[np.ndarray] = []

        for branch_id, branch in self._branch_id_to_branch.items():
            base_point_idx: int = len(points)
            points.extend(branch.medial_points)
            num_points: int = len(branch.medial_points)
            lines.extend(zip(
                np.arange(num_points - 1) + base_point_idx,
                np.arange(num_points - 1) + 1 + base_point_idx
            ))
            branch_colors: np.ndarray = np.zeros((num_points - 1, 3), dtype=float)
            if branch_id in optimized_branch_ids:
                branch_colors[0, :] = np.array([1., 0., 0.])
            colors.extend(branch_colors)
        
        lineset.points = o3d.utility.Vector3dVector(np.array(points))
        lineset.lines = o3d.utility.Vector2iVector(np.array(lines))
        lineset.colors = o3d.utility.Vector3dVector(np.array(colors))

        return f"Optimized furcations.", lineset
    
    def _interpolate_joint_points(self) -> None:
        branch_id_to_joint_point_offset: Dict[int, int] = {} # Fixed

        for branch_id, branch in self._branch_id_to_branch.items():
            branch_id_to_joint_point_offset[branch_id] = 0
            branch.joint_point_idx = branch.joint_point_idx + branch_id_to_joint_point_offset[branch.parent_id] if branch.joint_point_idx > 0 else branch.joint_point_idx
            
            if branch.parent_id == -1:
                continue    

            parent_branch: Branch = self._branch_id_to_branch[branch.parent_id]

            parent_direction: np.ndarray = normalize(
                parent_branch.medial_points[branch.joint_point_idx] - parent_branch.medial_points[branch.joint_point_idx - 1] \
                if branch.joint_point_idx > 0 else \
                parent_branch.medial_points[1] - parent_branch.medial_points[0]
            )

            medial_points: np.ndarray = branch.medial_points

            p0: np.ndarray = medial_points[0]
            p1: np.ndarray = medial_points[1]
            branch_direction: np.ndarray = normalize(medial_points[2] - medial_points[1] if len(medial_points) > 2 else medial_points[1] - medial_points[0])
            
            # Suppose there exists a point Q between P0 and P1 such that the direction of QP1 is the same as the direction of P1P2
            p0p1_norm: np.ndarray = np.linalg.norm(p1 - p0)

            joint_points: np.ndarray = hermite_curve(
                p0, p1, parent_direction * p0p1_norm * 0.33, branch_direction * p0p1_norm * 0.67, 
                num_points=self._num_hermite_nodes
            )
            branch.medial_points = np.vstack((
                joint_points,
                medial_points[2:]
            ))

            branch.radii = np.concatenate((
                np.linspace(branch.radii[0], branch.radii[1], len(joint_points)),
                branch.radii[2:]
            ))

            branch_id_to_joint_point_offset[branch_id] = len(joint_points) - 2
        return

    def _model(self) -> Optional[Tuple[str, o3d.geometry.TriangleMesh]]:
        max_branch_order: int = 0

        for branch in self._branch_id_to_branch.values():
            max_branch_order = max(max_branch_order, branch.order)

        branch_order_colors: List[Tuple[float, float, float]] = get_distinct_colors(max_branch_order + 1)

        for branch_id, branch in self._branch_id_to_branch.items():
            active_medial_point_start_idx: int = 0
            offset: int = 0 # Points interpolated by Hermite are not reduced by RDP to maintain visual effects

            # find the first point outside the parent branch mesh
            # This is the foundation of parameter calculation
            if branch.parent_id != -1:
                parent_branch_mesh: o3d.geometry.TriangleMesh = self._branch_id_to_branch[branch.parent_id].arterial_snake
                triangle_normals: np.ndarray = np.asarray(parent_branch_mesh.triangle_normals)
                parent_branch_mesh_t: o3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh.from_legacy(parent_branch_mesh)

                scene: o3d.t.geometry.RaycastingScene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(parent_branch_mesh_t)
                for i, medial_point in enumerate(branch.medial_points):
                    result: Dict[str, Any] = scene.compute_closest_points(
                        o3d.core.Tensor([medial_point.tolist()], dtype=o3d.core.Dtype.Float32)
                    )
                    closest_point: np.ndarray = result["points"].numpy()[0]
                    primitive_id: np.ndarray = int(result["primitive_ids"].numpy()[0])
                    if primitive_id >= len(triangle_normals):
                        continue
                    if np.dot(medial_point - closest_point, triangle_normals[primitive_id]) >= 0.0:
                        active_medial_point_start_idx = i
                        break   
                offset = self._num_hermite_nodes - 1

            branch.active_medial_point_start_idx = active_medial_point_start_idx
            
            # Apply Ramer–Douglas–Peucker polyline simplification to the mesh to export a much smaller model
            # BUT DO NOT MODIFY THE ORIGINAL PATH (the point at joint_point_idx might be missing).
            reserved_medial_points: np.ndarray
            reserved_medial_point_indices: np.ndarray
            reserved_medial_points, reserved_medial_point_indices = rdp_fast(branch.medial_points[offset:], self._rdp_epsilon)
            
            reserved_medial_points = np.concatenate((
                branch.medial_points[:offset], 
                reserved_medial_points
            ))
            reserved_radii: np.ndarray = np.concatenate((
                branch.radii[:offset], 
                branch.radii[reserved_medial_point_indices + offset]
            ))
            
            mesh: o3d.geometry.TriangleMesh = generate_arterial_snake(
                reserved_medial_points, 
                reserved_radii, 
                self._num_sectional_vertices
            )
            mesh.paint_uniform_color(np.array(branch_order_colors[branch.order]))
            self._branch_id_to_branch[branch_id].arterial_snake = mesh

            # Pruning measurement, consistent with the perspective of Forest mensuration, is only used for parameter calculation
            mesh2: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
            try:
                mesh2 = generate_arterial_snake(
                    branch.medial_points[active_medial_point_start_idx:], 
                    branch.radii[active_medial_point_start_idx:], 
                    self._num_sectional_vertices
                )
            except Exception:
                pass
            self._branch_id_to_branch[branch_id].backup_arterial_snake = mesh2

            self._branch_id_to_branch[branch_id].num_sectional_vertices = self._num_sectional_vertices

            branch.base_radius = None

        if not self._verbose:
            return
        
        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        for branch_id, branch in self._branch_id_to_branch.items():
            branch_mesh: o3d.geometry.TriangleMesh = branch.arterial_snake
            mesh += branch_mesh

        return f"Modeled.", mesh

    def run(self, branch_id_to_branch: Dict[int, Branch]) -> Generator[Any, None, Dict[int, Branch]]:
        self._branch_id_to_branch = branch_id_to_branch

        for fn in self._pipeline:
            yield fn()

        self._clear()
        return self._branch_id_to_branch
