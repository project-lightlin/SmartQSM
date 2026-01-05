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
import numpy as np
import pandas as pd
from utils.pandas_extra import append as dataframe_append
from typing import Dict, Set, Callable, Optional
from utils.arterial_snake import calculate_rough_volume, generate_arterial_snake
import open3d as o3d
from scipy.spatial import ConvexHull
from utils.numpy_extra import calculate_heading_angle, calculate_distances_from_points_to_line, calculate_direction_of_ordered_points, calculate_distances_from_points_to_line
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
import networkx as nx

def _sample_polyline(points, max_radius, step):
    points = np.asarray(points)
    if len(points) == 0:
        return None
    if len(points) == 1:
        return np.atleast_2d(points)
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist_along = np.cumsum(np.insert(seg_lengths, 0, 0.0))
    sample_dists = np.arange(0, max_radius + 1e-12, step)
    samples = []
    for sd in sample_dists:
        if sd > dist_along[-1]:
            break
        idx = np.searchsorted(dist_along, sd) - 1
        idx = max(idx, 0)
        seg_len = seg_lengths[idx]
        if seg_len == 0:
            samples.append(points[idx])
            continue
        t = (sd - dist_along[idx]) / seg_len
        p = points[idx] * (1 - t) + points[idx+1] * t
        samples.append(p)
    return np.array(samples)

class ParameterExtraction:
    _branch_id_to_branch: Dict[int, Branch]
    _points: np.ndarray
    _global_shift: np.ndarray
    tree_dataframe: pd.DataFrame
    branch_dataframe: pd.DataFrame
    crown_convex_hull: o3d.geometry.TriangleMesh
    active_crown_convex_hull: o3d.geometry.TriangleMesh
    _measurement_radius: float
    _sampling_size: float
    _tree_parameter_to_dtype: Dict[str, str]
    _branch_parameter_to_dtype: Dict[str, str]
    _degree_resolution: float
    _upper_bound_of_crown_radius: float

    def __init__(self, branch_id_to_branch: Dict[int, Branch], points: np.ndarray, global_shift: np.ndarray, measurement_radius: float = 0.05, sampling_size: float = 0.005, degree_resolution: float = 1.0, upper_bound_of_crown_radius: float = 1e4) -> None:
        self._branch_id_to_branch = branch_id_to_branch
        self._points = points
        self._global_shift = global_shift

        self._measurement_radius = measurement_radius
        self._sampling_size = sampling_size
        self._degree_resolution = degree_resolution
        self._upper_bound_of_crown_radius = upper_bound_of_crown_radius

        self._tree_parameter_to_dtype = {
            "X_m": "float64",
            "Y_m": "float64",
            "altitude_m": "float32",
            "num_branches": "uint64",
            "max_order": "uint64",
            "height_m": "float32",
            "DBH_cm": "float32",
            "ground_diameter_cm": "float32",
            "bole_height_m": "float32",
            "diameter_at_bole_height_cm": "float32",
            "bole_length_m": "float32",
            "bole_area_m2": "float32",
            "bole_volume_L": "float32",
            "trunk_length_m": "float32",
            "trunk_area_m2": "float32",
            "trunk_volume_L": "float32",
            "stem_length_m": "float32",
            "stem_area_m2": "float32",
            "stem_volume_L": "float32",
            "within_crown_stem_length_m": "float32",
            "within_crown_stem_area_m2": "float32",
            "within_crown_stem_volume_L": "float32",
            "min_crown_radius_m": "float32",
            "azimuth_of_min_crown_radius_deg": "float32",
            "height_at_min_crown_radius_m": "float32",
            "mean_crown_radius_m": "float32",
            "max_crown_radius_m": "float32",
            "azimuth_of_max_crown_radius_deg": "float32",
            "height_at_max_crown_radius_m": "float32",
            "min_crown_width_m": "float32",
            "azimuth_of_min_crown_width_deg": "float32",
            "mean_crown_width_m": "float32",
            "max_crown_width_m": "float32",
            "azimuth_of_max_crown_width_deg": "float32",
            "east_west_crown_width_m": "float32",
            "north_south_crown_width_m": "float32",
            "crown_convex_area_m2": "float32",
            "crown_convex_volume_L": "float32",
            "active_crown_convex_area_m2": "float32",
            "active_crown_convex_volume_L": "float32",
            "crown_projection_convex_area_m2": "float32",
            "crown_perimeter_m": "float32",
            "canopy_area_m2": "float32",
            "crown_center_offset_m": "float32",
            "crown_center_azimuth_deg": "float32",
            "min_crown_spread_m": "float32",
            "azimuth_of_min_crown_spread_deg": "float32",
            "max_crown_spread_m": "float32",
            "azimuth_of_max_crown_spread_deg": "float32",
            "mid_length_diameter_cm": "float32",
            "tip_diameter_cm": "float32",
            "max_spread_m": "float32",
            "azimuth_deg": "float32",
            "zenith_deg": "float32",
            "chord_length_m": "float32",
            "arc_height_m": "float32",
            "height_difference_m": "float32",
            "tip_based_DINC_m": "float32",
            "apex_based_DINC_m": "float32",
        }
        self.tree_dataframe = pd.DataFrame({k: pd.Series() for k in self._tree_parameter_to_dtype.keys()})
        self._branch_parameter_to_dtype = {
            "id": "uint64",
            "parent": "uint64",
            "order": "uint64",
            "base_height_m": "float32",
            "base_diameter_cm": "float32",
            "mid_length_diameter_cm": "float32",
            "tip_diameter_cm": "float32",
            "length_m": "float32",
            "area_m2": "float32",
            "volume_L": "float32",
            "max_spread_m": "float32",
            "azimuth_deg": "float32",
            "zenith_deg": "float32",
            "chord_length_m": "float32",
            "arc_height_m": "float32",
            "height_difference_m": "float32",
            "branching_radius_m": "float32",
            "branching_angle_deg": "float32",
            "tip_deflection_angle_deg": "float32",
            "vertical_deflection_angle_deg": "float32",
            "tip_based_DINC_m": "float32",
            "apex_based_DINC_m": "float32",
            "growth_length_m": "float32",
            "growth_area_m2": "float32",
            "growth_volume_L": "float32",
            "base_offset_m": "float32",
            "base_azimuth_deg": "float32",
            "insertion_distance_m": "float32",
            "is_a_main_branch": "bool",
        }
        self.branch_dataframe = pd.DataFrame({k: pd.Series() for k in self._branch_parameter_to_dtype.keys()})
        
        self._initialize()
        self._extract()
        self.tree_dataframe = self.tree_dataframe.astype(self._tree_parameter_to_dtype)
        self.branch_dataframe = self.branch_dataframe.astype(self._branch_parameter_to_dtype)
        return
    
    def _initialize(self) -> None:
        dataframe_append(self.tree_dataframe)
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1: # Trunk
                continue
            id: int = dataframe_append(self.branch_dataframe)
            self.branch_dataframe.loc[id, "id"] = branch_id
            self.branch_dataframe.loc[id, "parent"] = branch.parent_id
        return
    
    def _extract(self) -> None:
        # General
        trunk: Branch = self._branch_id_to_branch[1]
        tree_position: np.ndarray = trunk.medial_points[0][:2]
        z_min: float = np.min(self._points[:, 2])
        tree_height: float = np.max(self._points[:, 2]) - z_min
        self.tree_dataframe.loc[0, "X_m"] = tree_position[0] - self._global_shift[0]
        self.tree_dataframe.loc[0, "Y_m"] = tree_position[1] - self._global_shift[1]
        self.tree_dataframe.loc[0, "altitude_m"] = -self._global_shift[2] + z_min
        self.tree_dataframe.loc[0, "num_branches"] = len(self._branch_id_to_branch)
        self.tree_dataframe.loc[0, "height_m"] = tree_height

        # Trunk diameters & length
        ground_diameter: float = 0.
        dbh: float = 0.
        line_lengths: np.ndarray = np.linalg.norm(trunk.medial_points[1:] - trunk.medial_points[:-1], axis=-1)
        cumulative_lengths_of_points: np.ndarray = np.concatenate([[0], np.cumsum(line_lengths)])
        mid_length_diameter: float = 0.
        for i in range(len(trunk.medial_points) - 1):
            p_bottom: np.ndarray = trunk.medial_points[i]
            r_bottom: float = trunk.radii[i]
            p_top: np.ndarray = self._branch_id_to_branch[1].medial_points[i + 1]
            r_top: float = trunk.radii[i + 1]
            if p_bottom[2] <= 0.2 + z_min and p_top[2] >= 0.2 + z_min and ground_diameter == 0.: # Measure the breast height diameter at 0.2 m
                ground_diameter = 2. * (r_top + (r_bottom - r_top) * (p_top[2] - 0.2 - z_min) / (p_top[2] - p_bottom[2]))
            if p_bottom[2] <= 1.3 + z_min and p_top[2] >= 1.3 + z_min and dbh == 0.: # Measure the breast height diameter at 1.3 m
                dbh = 2. * (r_top + (r_bottom - r_top) * (p_top[2] - 1.3 - z_min) / (p_top[2] - p_bottom[2]))
            if cumulative_lengths_of_points[i] <= cumulative_lengths_of_points[-1] / 2. and cumulative_lengths_of_points[i + 1] >= cumulative_lengths_of_points[-1] / 2.:
                mid_length_diameter = 2. * (
                    r_top + (r_bottom - r_top) * (cumulative_lengths_of_points[i + 1] - cumulative_lengths_of_points[-1] / 2.) / (cumulative_lengths_of_points[i + 1] - cumulative_lengths_of_points[i])
                )
        if ground_diameter == 0.:
            ground_diameter = 2. * self._branch_id_to_branch[1].radii[0]
        trunk_length: float = cumulative_lengths_of_points[-1]
        self.tree_dataframe.loc[0, "ground_diameter_cm"] = ground_diameter * 100.
        self.tree_dataframe.loc[0, "DBH_cm"] = dbh * 100.
        self.tree_dataframe.loc[0, "mid_length_diameter_cm"] = mid_length_diameter * 100.
        self.tree_dataframe.loc[0, "tip_diameter_cm"] = trunk.radii[-1] * 2. * 100.
        self.tree_dataframe.loc[0, "trunk_length_m"] = trunk_length
        stem_length: float = trunk_length

        # Bole height; max branch order
        # Branch order & length & diameters; base height; main branch marker
        min_base_height: float = np.inf
        bole_height: float = np.inf
        max_branch_order: int = 0
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1: # Trunk
                continue

            base_height: float 
            d_base: float 
            try:
                d_base = branch.radii[branch.active_medial_point_start_id] * 2.
                p_base: np.ndarray = branch.medial_points[branch.active_medial_point_start_id]
                p_start: np.ndarray = branch.medial_points[branch.active_medial_point_start_id + 1]
                base_to_start_direction: np.ndarray = p_start - p_base
                base_to_start_direction /= np.linalg.norm(base_to_start_direction)
                base_height = branch.medial_points[branch.active_medial_point_start_id][2] - (d_base / 2.) * np.sqrt(1. - np.clip(np.dot(base_to_start_direction, [0., 0., 1.]).item() ** 2., 0., 1.)) - z_min
            except Exception:
                d_base = branch.radii[-1] * 2.
                p_base: np.ndarray = branch.medial_points[-2]
                p_start: np.ndarray = branch.medial_points[-1]
                base_to_start_direction: np.ndarray = p_start - p_base
                base_to_start_direction /= np.linalg.norm(base_to_start_direction)
                base_height = branch.medial_points[-1][2] - (d_base / 2.) * np.sqrt(1. - np.clip(np.dot(base_to_start_direction, [0., 0., 1.]).item() ** 2., 0., 1.)) - z_min
                pass
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "base_height_m"] = base_height
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "base_diameter_cm"] = d_base * 100.
            
            min_base_height = min(min_base_height, base_height)
            branch_order: int = branch.order
            max_branch_order = max(max_branch_order, branch_order)
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "order"] = branch_order
            is_a_main_branch: Callable[[], bool] = lambda: d_base >= 0.2 * (dbh if branch_order == 1 else self.branch_dataframe.loc[self.branch_dataframe["id"] == branch.parent_id, "base_diameter_cm"].item() / 100)
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "is_a_main_branch"] = is_a_main_branch()
            if branch_order == 1 and is_a_main_branch():
                bole_height = min(bole_height, base_height)
            
            cumulative_lengths_of_points: np.ndarray = np.array([0.0])
            mid_length_diameter: float = 2. * branch.radii[-1]
            try:
                line_lengths = np.linalg.norm(branch.medial_points[branch.active_medial_point_start_id + 1:] - branch.medial_points[branch.active_medial_point_start_id:-1], axis=-1)
                cumulative_lengths_of_points = np.concatenate([[0.0], np.cumsum(line_lengths)])
                for j in range(len(branch.medial_points) - branch.active_medial_point_start_id - 1):
                    r_bottom: float = branch.radii[branch.active_medial_point_start_id + j]
                    r_top: float = branch.radii[branch.active_medial_point_start_id + j + 1]
                    if cumulative_lengths_of_points[j] <= cumulative_lengths_of_points[-1] / 2. and cumulative_lengths_of_points[j + 1] >= cumulative_lengths_of_points[-1] / 2.:
                        mid_length_diameter = 2. * (
                            r_top + (r_bottom - r_top) * (cumulative_lengths_of_points[j + 1] - cumulative_lengths_of_points[-1] / 2.) / (cumulative_lengths_of_points[j + 1] - cumulative_lengths_of_points[j])
                        )
            except Exception:
                pass
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "mid_length_diameter_cm"] = mid_length_diameter * 100.
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "tip_diameter_cm"] = branch.radii[-1] * 2. * 100.
            branch_length: float = cumulative_lengths_of_points[-1]
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "length_m"] = branch_length
            stem_length += branch_length

        if bole_height == np.inf:
            bole_height = min_base_height if min_base_height != np.inf else 0.0
        self.tree_dataframe.loc[0, "bole_height_m"] = bole_height
        self.tree_dataframe.loc[0, "max_order"] = max_branch_order
        self.tree_dataframe.loc[0, "stem_length_m"] = stem_length
        
        # Trunk / branch / stem area & volume
        trunk_area: float = trunk.backup_arterial_snake.get_surface_area()
        trunk_volume: float = 0.0
        try:
            trunk_volume = calculate_rough_volume(trunk.backup_arterial_snake, trunk.num_sectional_vertices) 
        except Exception:
            pass
        self.tree_dataframe.loc[0, "trunk_area_m2"] = trunk_area
        self.tree_dataframe.loc[0, "trunk_volume_L"] = trunk_volume * 1000.
        stem_area: float = trunk_area
        stem_volume: float = trunk_volume
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1: # Trunk
                continue
            branch_area: float = branch.backup_arterial_snake.get_surface_area()
            branch_volume: float = 0.0
            try:
                branch_volume = calculate_rough_volume(branch.backup_arterial_snake, branch.num_sectional_vertices) 
            except Exception:
                pass
            stem_area += branch_area
            stem_volume += branch_volume
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "area_m2"] = branch_area
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "volume_L"] = branch_volume * 1000.
        self.tree_dataframe.loc[0, "stem_area_m2"] = stem_area
        self.tree_dataframe.loc[0, "stem_volume_L"] = stem_volume * 1000.
        
        # Locating the bole top
        bole_top_id: int = 0
        for i in range(len(trunk.medial_points) - 1):
            if trunk.medial_points[i][2] <= bole_height + z_min and trunk.medial_points[i + 1][2] >= bole_height + z_min:
                bole_top_id = i
                break
        within_crown_branch_ids: Set[int] = set()
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1: # Trunk
                continue
            if branch.order == 1:
                if self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "base_height_m"].item() >= bole_height:
                    within_crown_branch_ids.add(branch_id)
            else:
                if branch.parent_id in within_crown_branch_ids:
                    within_crown_branch_ids.add(branch_id)
        
        # Bole parameters (except bole height)
        bole_length: float = 0.
        bole_area: float = 0.
        bole_volume: float = 0.
        active_crown_vertex_start_in_trunk_mesh: int = 0
        if bole_top_id > 0:
            for i in range(bole_top_id):
                bole_length += np.linalg.norm(trunk.medial_points[i + 1] - trunk.medial_points[i])
            bole_mesh: o3d.geometry.TriangleMesh = generate_arterial_snake(trunk.medial_points[:bole_top_id + 1], trunk.radii[:bole_top_id + 1], trunk.num_sectional_vertices)
            active_crown_vertex_start_in_trunk_mesh = len(bole_mesh.vertices)
            bole_area = bole_mesh.get_surface_area()
            bole_volume = calculate_rough_volume(bole_mesh, trunk.num_sectional_vertices)
        self.tree_dataframe.loc[0, "bole_length_m"] = bole_length
        self.tree_dataframe.loc[0, "bole_area_m2"] = bole_area
        self.tree_dataframe.loc[0, "bole_volume_L"] = bole_volume * 1000.
        self.tree_dataframe.loc[0, "diameter_at_bole_height_cm"] = trunk.radii[bole_top_id] * 2. * 100.

        # Active crown (based on mesh)
        active_crown_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        active_crown_cloud.points = o3d.utility.Vector3dVector(np.asarray(trunk.arterial_snake.vertices)[active_crown_vertex_start_in_trunk_mesh:])

        # Within-crown stem parameters
        within_crown_branch_length: float = 0.
        within_crown_branch_area: float = 0.
        within_crown_branch_volume_L: float = 0.
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1:
                continue
            if branch_id in within_crown_branch_ids:
                within_crown_branch_length += self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "length_m"].item()
                within_crown_branch_area += self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "area_m2"].item()
                within_crown_branch_volume_L += self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "volume_L"].item()
                branch_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
                branch_cloud.points = o3d.utility.Vector3dVector(np.asarray(branch.arterial_snake.vertices))
                active_crown_cloud += branch_cloud
        self.tree_dataframe.loc[0, "within_crown_stem_length_m"] = within_crown_branch_length + self.tree_dataframe.loc[0, "trunk_length_m"] - self.tree_dataframe.loc[0, "bole_length_m"]
        self.tree_dataframe.loc[0, "within_crown_stem_area_m2"] = within_crown_branch_area + self.tree_dataframe.loc[0, "trunk_area_m2"] - self.tree_dataframe.loc[0, "bole_area_m2"]
        self.tree_dataframe.loc[0, "within_crown_stem_volume_L"] = within_crown_branch_volume_L  + self.tree_dataframe.loc[0, "trunk_volume_L"] - self.tree_dataframe.loc[0, "bole_volume_L"]
        
        # Canopy area (not only including the crown projection)
        canopy_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id in within_crown_branch_ids:
                canopy_mesh += branch.arterial_snake
        mesh_vertices: np.ndarray = np.asarray(canopy_mesh.vertices)
        mesh_vertices[:, 2] = 0 
        canopy_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
        canopy_mesh.cluster_connected_triangles()
        canopy_mesh.remove_duplicated_vertices()
        canopy_mesh.remove_duplicated_triangles()
        self.tree_dataframe.loc[0, "canopy_area_m2"] = canopy_mesh.get_surface_area()

        # Crown (project) convex hull parameters
        crown_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        crown_cloud.points = o3d.utility.Vector3dVector(self._points[
            self._points[:, 2] >= bole_height + z_min
        ])
        crown_points: np.ndarray = np.asarray(crown_cloud.points)

        self.crown_convex_hull = crown_cloud.compute_convex_hull()[0]
        crown_convex_hull: ConvexHull = ConvexHull(np.asarray(self.crown_convex_hull.vertices))
        self.tree_dataframe.loc[0, "crown_convex_area_m2"] =  crown_convex_hull.area
        self.tree_dataframe.loc[0, "crown_convex_volume_L"] = crown_convex_hull.volume * 1000.
        
        crown_projection_convex_area: float = 0.0
        crown_perimeter: float = 0.0
        crown_projection_points: np.ndarray = np.delete(crown_points, 2, axis=1)
        try:
            crown_projection_convex_hull: ConvexHull = ConvexHull(crown_projection_points)
            crown_projection_convex_area = crown_projection_convex_hull.volume # 2D volume is the area
            crown_perimeter = crown_projection_convex_hull.area # 2D area is the perimeter
        except Exception:
            pass
        self.tree_dataframe.loc[0, "crown_projection_convex_area_m2"] = crown_projection_convex_area
        self.tree_dataframe.loc[0, "crown_perimeter_m"] = crown_perimeter

        crown_center: np.ndarray = (crown_projection_points.min(axis=0) + crown_projection_points.max(axis=0)) / 2.
        crown_projection_convex_polygon_vertices: np.ndarray = crown_projection_points[crown_projection_convex_hull.vertices]
        crown_projection_convex_polygon: Polygon = Polygon(crown_projection_convex_polygon_vertices)
        
        # Active crown parameters
        activate_crown_convex_hull_area: float = 0.0
        activate_crown_convex_hull_volume: float = 0.0
        try:
            self.active_crown_convex_hull = active_crown_cloud.compute_convex_hull()[0]
            activate_crown_convex_hull: ConvexHull = ConvexHull(np.asarray(self.active_crown_convex_hull.vertices))
            activate_crown_convex_hull_area = activate_crown_convex_hull.area
            activate_crown_convex_hull_volume = activate_crown_convex_hull.volume 
        except Exception:
            pass
        self.tree_dataframe.loc[0, "active_crown_convex_area_m2"] = activate_crown_convex_hull_area
        self.tree_dataframe.loc[0, "active_crown_convex_volume_L"] = activate_crown_convex_hull_volume * 1000. 

        # Crown width
        self.tree_dataframe.loc[0, "east_west_crown_width_m"] = crown_projection_points[:, 0].max() - crown_projection_points[:, 0].min()
        self.tree_dataframe.loc[0, "north_south_crown_width_m"] = crown_projection_points[:, 1].max() - crown_projection_points[:, 1].min()
        
        min_crown_width: float = np.inf
        azimuth_of_min_crown_width: float = 0.
        mean_crown_width: float = 0.
        max_crown_width: float = -np.inf
        azimuth_of_max_crown_width: float = 0.
        for i in range(0, int(180. / self._degree_resolution)):
            angle: float = i * self._degree_resolution
            dx: float = np.cos(np.deg2rad(angle))
            dy: float = np.sin(np.deg2rad(angle))

            projections: np.ndarray = (crown_projection_points - crown_center) @ np.array([dx, dy])
            crown_width = projections.max() - projections.min()

            azimuth: float = 90 - angle if angle <= 90 else 270 - angle
            if crown_width < min_crown_width:
                min_crown_width = crown_width
                azimuth_of_min_crown_width = azimuth
            if crown_width > max_crown_width:
                max_crown_width = crown_width
                azimuth_of_max_crown_width = azimuth

            mean_crown_width += crown_width
        mean_crown_width /= int(180. / self._degree_resolution)
        self.tree_dataframe.loc[0, "min_crown_width_m"] = min_crown_width
        self.tree_dataframe.loc[0, "azimuth_of_min_crown_width_deg"] = azimuth_of_min_crown_width
        self.tree_dataframe.loc[0, "mean_crown_width_m"] = mean_crown_width
        self.tree_dataframe.loc[0, "max_crown_width_m"] = max_crown_width
        self.tree_dataframe.loc[0, "azimuth_of_max_crown_width_deg"] = azimuth_of_max_crown_width
        
        # Crown radius
        intervalwise_crown_radii: np.ndarray = np.zeros(36, dtype=np.float32)
        azimuths_of_intervalwise_crown_radius: np.ndarray = np.zeros(36, dtype=np.float32)
        heights_at_intervalwise_crown_radius: np.ndarray = np.zeros(36, dtype=np.float32)

        for i in range(crown_projection_points.shape[0]):
            azimuth: float = calculate_heading_angle(crown_projection_points[i] - crown_center)
            max_crown_radius_in_interval: float = intervalwise_crown_radii[int(azimuth // 10)]
            centrifugal_distance: float = np.linalg.norm(crown_projection_points[i] - crown_center)
            if max_crown_radius_in_interval < centrifugal_distance:
                azimuths_of_intervalwise_crown_radius[int(azimuth // 10)] = azimuth
                intervalwise_crown_radii[int(azimuth // 10)] = centrifugal_distance
                heights_at_intervalwise_crown_radius[int(azimuth // 10)] = crown_points[i, 2] - z_min
        interval_id_of_min_crown_radius: int = np.argmin(intervalwise_crown_radii)
        interval_id_of_max_crown_radius: int = np.argmax(intervalwise_crown_radii)
        self.tree_dataframe.loc[0, "min_crown_radius_m"] = intervalwise_crown_radii[interval_id_of_min_crown_radius]
        self.tree_dataframe.loc[0, "azimuth_of_min_crown_radius_deg"] = azimuths_of_intervalwise_crown_radius[interval_id_of_min_crown_radius]
        self.tree_dataframe.loc[0, "height_at_min_crown_radius_m"] = heights_at_intervalwise_crown_radius[interval_id_of_min_crown_radius]
        self.tree_dataframe.loc[0, "mean_crown_radius_m"] = intervalwise_crown_radii.mean()
        self.tree_dataframe.loc[0, "max_crown_radius_m"] = intervalwise_crown_radii[interval_id_of_max_crown_radius]
        self.tree_dataframe.loc[0, "azimuth_of_max_crown_radius_deg"] = azimuths_of_intervalwise_crown_radius[interval_id_of_max_crown_radius]
        self.tree_dataframe.loc[0, "height_at_max_crown_radius_m"] = heights_at_intervalwise_crown_radius[interval_id_of_max_crown_radius]

        # Tree center-based 2D crown parameters (including offset and spread)
        tree_position_point: Point = Point(tree_position)
        crown_center_coordinate_offsets: np.ndarray = crown_center - tree_position
        self.tree_dataframe.loc[0, "crown_center_offset_m"] = np.linalg.norm(crown_center_coordinate_offsets)
        self.tree_dataframe.loc[0, "crown_center_azimuth_deg"] = calculate_heading_angle(crown_center_coordinate_offsets)
        nearest_crown_boundary_point: Point = nearest_points(tree_position_point, crown_projection_convex_polygon.boundary)[1]
        self.tree_dataframe.loc[0, "min_crown_spread_m"] = tree_position_point.distance(nearest_crown_boundary_point) # The min distance between a point and a polygon should be a point-line or point-point distance
        self.tree_dataframe.loc[0, "azimuth_of_min_crown_spread_deg"] = calculate_heading_angle(np.array([nearest_crown_boundary_point.x - tree_position_point.x, nearest_crown_boundary_point.y - tree_position_point.y]))
        centrifugal_distances_to_crown_projection_convex_polygon_vertices: np.ndarray = np.linalg.norm(crown_projection_convex_polygon_vertices - tree_position, axis=1)
        farthest_crown_projection_convex_polygon_vertex_id: int = np.argmax(centrifugal_distances_to_crown_projection_convex_polygon_vertices)
        self.tree_dataframe.loc[0, "max_crown_spread_m"] = centrifugal_distances_to_crown_projection_convex_polygon_vertices[farthest_crown_projection_convex_polygon_vertex_id] ## The min distance between a point and a polygon must be a point-point distance
        self.tree_dataframe.loc[0, "azimuth_of_max_crown_spread_deg"] = calculate_heading_angle(crown_projection_convex_polygon_vertices[farthest_crown_projection_convex_polygon_vertex_id] - tree_position) 

        # Azimuth & zenith & max horizontal extension & chord length & arc height
        trunk_displacement: np.ndarray = trunk.medial_points[-1] - trunk.medial_points[0]
        trunk_chord_length: float = np.linalg.norm(trunk_displacement)
        self.tree_dataframe.loc[0, "max_spread_m"] = np.linalg.norm(trunk.medial_points[1:, :2] - trunk.medial_points[0, :2], axis=-1).max()
        self.tree_dataframe.loc[0, "azimuth_deg"] = calculate_heading_angle(trunk_displacement[:2]) 
        self.tree_dataframe.loc[0, "zenith_deg"] = np.rad2deg(np.arccos(np.clip(np.dot(np.array([0., 0., 1.]), trunk_displacement / trunk_chord_length).item(), -1.0, 1.0)))
        self.tree_dataframe.loc[0, "chord_length_m"] = trunk_chord_length
        self.tree_dataframe.loc[0, "arc_height_m"] = np.max(calculate_distances_from_points_to_line(trunk.medial_points, trunk.medial_points[0], trunk.medial_points[-1]))
        for branch_id, branch in self._branch_id_to_branch.items():
            branch_chord_length: float = 0.0
            max_spread: float = 0.0
            azimuth_angle: float = 0.0
            arc_height: float = 0.0
            zenith_angle: float = 0.0
            base_offset: float = 0.0
            base_azimuth_angle: float = 0.0
            try:
                branch_displacement: np.ndarray = branch.medial_points[-1] - branch.medial_points[branch.active_medial_point_start_id]
                branch_chord_length = np.linalg.norm(branch_displacement)
                azimuth_angle = calculate_heading_angle(branch_displacement[:2]) 
                zenith_angle = np.rad2deg(np.arccos(np.clip(np.dot(np.array([0., 0., 1.]), branch_displacement / (branch_chord_length + 1e-6)).item(), -1.0, 1.0)))
                arc_height = np.max(calculate_distances_from_points_to_line(branch.medial_points, branch.medial_points[branch.active_medial_point_start_id], branch.medial_points[-1]))
                base_offset = np.linalg.norm(branch.medial_points[branch.active_medial_point_start_id][:2] - tree_position)
                base_azimuth_angle = calculate_heading_angle(branch.medial_points[branch.active_medial_point_start_id][:2] - tree_position)
                max_spread = np.linalg.norm(branch.medial_points[branch.active_medial_point_start_id + 1:, :2] - branch.medial_points[branch.active_medial_point_start_id, :2], axis=-1).max()
            except Exception:
                pass
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "max_spread_m"] = max_spread
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "azimuth_deg"] = azimuth_angle
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "zenith_deg"] = zenith_angle
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "chord_length_m"] = branch_chord_length
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "arc_height_m"] = arc_height
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "base_offset_m"] = base_offset
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "base_azimuth_deg"] = base_azimuth_angle
        
        # Height difference & branching radius & branching angle & deflection angle of the branch tip & vertical deflection angle & tip_based_DINC & apex_based_DINC
        segmentwise_directions: np.ndarray = trunk.medial_points[1:] - trunk.medial_points[:-1]
        segmentwise_directions: np.ndarray = segmentwise_directions / np.linalg.norm(segmentwise_directions, axis=-1, keepdims=True)
        segmentwise_directions = np.vstack([segmentwise_directions, segmentwise_directions[-1]])
        segmentwise_vertical_offsets: np.ndarray = trunk.radii * np.sqrt(1 - np.clip(np.dot(segmentwise_directions, np.array([0., 0., 1.])) ** 2,  0., 1.))
        upper_bounds: np.ndarray = trunk.medial_points[:, 2] + segmentwise_vertical_offsets
        lower_bounds: np.ndarray = trunk.medial_points[:, 2] - segmentwise_vertical_offsets
        self.tree_dataframe.loc[0, "height_difference_m"] = np.max(upper_bounds) - np.min(lower_bounds)
        self.tree_dataframe.loc[0, "tip_based_DINC_m"] = tree_height - upper_bounds[-1] + z_min
        self.tree_dataframe.loc[0, "apex_based_DINC_m"] = tree_height - np.max(upper_bounds) + z_min


        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1:
                continue

            height_difference: float = 0.0
            try:
                segmentwise_directions: np.ndarray = branch.medial_points[branch.active_medial_point_start_id + 1:] - branch.medial_points[branch.active_medial_point_start_id:-1]
                segmentwise_directions: np.ndarray = segmentwise_directions / np.linalg.norm(segmentwise_directions, axis=-1, keepdims=True)
                segmentwise_directions = np.vstack([segmentwise_directions, segmentwise_directions[-1]])
                segmentwise_vertical_offsets: np.ndarray = branch.radii[branch.active_medial_point_start_id:] * np.sqrt(1 - np.clip(np.dot(segmentwise_directions, np.array([0., 0., 1.])) ** 2,  0., 1.))
                upper_bounds: np.ndarray = branch.medial_points[branch.active_medial_point_start_id:, 2] + segmentwise_vertical_offsets
                lower_bounds: np.ndarray = branch.medial_points[branch.active_medial_point_start_id:, 2] - segmentwise_vertical_offsets
                height_difference = np.max(upper_bounds) - np.min(lower_bounds)
            except Exception:
                pass
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "height_difference_m"] = height_difference

            parent_branch: Branch = self._branch_id_to_branch[branch.parent_id]
            joint_point: np.ndarray = parent_branch.medial_points[branch.joint_point_id]
            parent_branch_samples: np.ndarray = _sample_polyline(parent_branch.medial_points[branch.joint_point_id:], self._measurement_radius, self._sampling_size)
            parent_local_branch_direction: np.ndarray = np.array([0., 0., 0.])
            if parent_branch_samples.shape[0] > 1:
                parent_local_branch_direction = calculate_direction_of_ordered_points(parent_branch_samples)
                parent_local_branch_direction /= np.linalg.norm(parent_local_branch_direction)
            current_local_branch_direction: np.ndarray = np.array([0., 0., 0.])
            try:
                current_branch_samples: np.ndarray = _sample_polyline(branch.medial_points[branch.active_medial_point_start_id:], self._measurement_radius, self._sampling_size)
                if current_branch_samples.shape[0] > 1:
                    current_local_branch_direction = calculate_direction_of_ordered_points(current_branch_samples)
                    current_local_branch_direction /= np.linalg.norm(current_local_branch_direction)
            except Exception:
                pass

            branching_angle: float = 0.
            tip_deflection_angle: float = 0.
            branching_radius: float = 0.
            vertical_deflection_angle: float = 0.
            if np.linalg.norm(parent_local_branch_direction) != 0.:
                if np.linalg.norm(current_local_branch_direction) != 0.:
                    branching_angle = np.rad2deg(np.arccos(np.clip(np.dot(parent_local_branch_direction, current_local_branch_direction).item(), -1.0, 1.0)))
                    vertical_deflection_angle = np.rad2deg(np.arccos(np.clip(np.dot(np.array([0., 0., 1.]), current_local_branch_direction).item(), -1.0, 1.0)))

                try:
                    arc_direction: np.ndarray = branch.medial_points[-1] - branch.medial_points[branch.active_medial_point_start_id]
                    if np.linalg.norm(arc_direction) != 0.:
                        arc_direction /= np.linalg.norm(arc_direction)
                        tip_deflection_angle = np.rad2deg(np.arccos(np.clip(np.dot(parent_local_branch_direction, arc_direction).item(), -1.0, 1.0)))
                    branching_radius = np.max(
                        calculate_distances_from_points_to_line(branch.medial_points[branch.active_medial_point_start_id:], joint_point, joint_point + parent_local_branch_direction)
                    )
                except Exception:
                    pass

            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "branching_angle_deg"] = branching_angle
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "tip_deflection_angle_deg"] = tip_deflection_angle
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "branching_radius_m"] = branching_radius
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "vertical_deflection_angle_deg"] = vertical_deflection_angle

            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "tip_based_DINC_m"] = tree_height - upper_bounds[-1] + z_min
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "apex_based_DINC_m"] = tree_height - np.max(upper_bounds) + z_min
        
        # Growth length, area, volume
        branch_id_to_growth_length: Dict[int, float] = {}
        branch_id_to_growth_area: Dict[int, float] = {}
        branch_id_to_growth_volume: Dict[int, float] = {}
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1:
                continue
            growth_length: float = self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "length_m"].item()
            growth_area: float = self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "area_m2"].item()
            growth_volume: float = self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "volume_L"].item()
            
            branch_id_to_growth_length[branch_id] = growth_length
            branch_id_to_growth_area[branch_id] = growth_area
            branch_id_to_growth_volume[branch_id] = growth_volume
            
            ancestor_branch_id: int = branch.parent_id
            while ancestor_branch_id != 1:
                branch_id_to_growth_length[ancestor_branch_id] += growth_length
                branch_id_to_growth_area[ancestor_branch_id] += growth_area
                branch_id_to_growth_volume[ancestor_branch_id] += growth_volume
                ancestor_branch_id = self._branch_id_to_branch[ancestor_branch_id].parent_id
        for branch_id, growth_length in branch_id_to_growth_length.items():
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "growth_length_m"] = growth_length
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "growth_area_m2"] = branch_id_to_growth_area[branch_id]
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "growth_volume_L"] = branch_id_to_growth_volume[branch_id]

        # insertion_distance
        for branch_id, branch in self._branch_id_to_branch.items():
            if branch_id == 1:
                continue
            insertion_distance: float = 0.
            parent_branch_id: int = branch.parent_id
            joint_point_id: int = branch.joint_point_id
            while parent_branch_id > 0:
                parent_branch: Branch = self._branch_id_to_branch[parent_branch_id]
                try:
                    polyline: np.ndarray = np.array(parent_branch.medial_points[parent_branch.active_medial_point_start_id:joint_point_id+1])
                    if len(polyline) > 2:
                        diffs = polyline[1:] - polyline[:-1]
                        insertion_distance += np.linalg.norm(diffs, axis=1).sum()
                except Exception:
                    pass
                parent_branch_id: int = parent_branch.parent_id
                joint_point_id: int = parent_branch.joint_point_id
            self.branch_dataframe.loc[self.branch_dataframe["id"] == branch_id, "insertion_distance_m"] = insertion_distance
        return