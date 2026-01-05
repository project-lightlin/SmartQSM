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

from typing import Dict, Tuple, List, Any, Optional, Union
from .data_type import Branch
import numpy as np
import open3d as o3d
import os
from utils.io3d import write_polyline
from scipy.io import savemat
from utils.hash import get_md5_hash
from .parameter_extraction import ParameterExtraction

def parameterize_and_export(
        branch_id_to_branch: Dict[int, Branch],
        points: np.ndarray,
        output_path_prefix: str,
        global_shift: np.ndarray,
        projection: str = "",
        params_for_parameter_extraction: Dict[str, Any] = {},
        creator: str = "SmartQSM",
) -> None:
    # Check order
    for branch_id, branch in branch_id_to_branch.items():
        if branch_id == 1:
            if branch.order != 0:
                raise ValueError(f"Branch 1 must be at order 0 to serve as the sole trunk of the tree, but got {branch.order}")
            continue
        ancestor_branch_id: int = branch.parent_id
        branch_order: int = 1
        while ancestor_branch_id != 1:
            branch_order += 1
            if ancestor_branch_id not in branch_id_to_branch:
                raise ValueError(f"Found a nonexistent parent branch {ancestor_branch_id}.")
            ancestor_branch_id = branch_id_to_branch[ancestor_branch_id].parent_id
        if branch_order != branch.order:
            raise ValueError(f"The actual order of branch {branch_id} is {branch_order}, but the property 'order' is {branch.order}")
    # Keep order non-decreasing, otherwise the growth length will be wrong.
    branch_id_to_branch = dict(sorted(branch_id_to_branch.items(), key=lambda item: item[1].order))

    os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    branch_mesh_path: str = f"{output_path_prefix}_branches.ply"
    crown_mesh_path: str = f"{output_path_prefix}_crown.ply"
    active_crown_mesh_path: str = f"{output_path_prefix}_active_crown.ply"
    skeleton_path: str = f"{output_path_prefix}_skeleton.dxf"
    qsm_path: str = f"{output_path_prefix}_qsm.mat"

    skeleton: o3d.geometry.LineSet = o3d.geometry.LineSet()
    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
    branch_id_to_triangle_count: Dict[int, int] = {}
    for branch_id, branch in branch_id_to_branch.items():
        mesh += branch.arterial_snake
        branch_id_to_triangle_count[branch_id] = len(branch.arterial_snake.triangles)
        medial_points: np.ndarray = branch.medial_points
        medial_axis: o3d.geometry.LineSet = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(medial_points),
            lines=o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(len(medial_points) - 1)]))
        )
        skeleton += medial_axis
    skeleton.translate(-global_shift)
    mesh.translate(-global_shift)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(branch_mesh_path, mesh, write_vertex_normals=True)
    branch_mesh_hash: str = get_md5_hash(branch_mesh_path)
    write_polyline(skeleton, skeleton_path)

    branch_id_to_triangle_id_range: Dict[int, Tuple[int, int]] = {}
    triangle_ids: List[int] = np.cumsum(list(branch_id_to_triangle_count.values())).tolist()
    for id, branch_id in enumerate(branch_id_to_branch.keys()):
        branch_id_to_triangle_id_range[branch_id] = (triangle_ids[id - 1] if id > 0 else 0, triangle_ids[id] - 1)

    # Compatible with TreeQSM
    qsm_cylinder_dict: Dict[str, Union[List[Any], np.ndarray]] = {
        "radius": [],
        "length": [],
        "start": [],
        "axis": [],
        "parent": [],
        "extension": [],
        "added": [],
        "UnmodRadius": [],
        "branch": [],
        "BranchOrder": [],
        "PositionInBranch": []
    }
    branch_id_to_start_cylinder_id: Dict[int, int] = {}
    added_cylinder_count: int = 0

    for branch_id, branch in branch_id_to_branch.items():
        medial_points: np.ndarray = branch.medial_points
        start: np.ndarray = medial_points[:-1].copy()
        end: np.ndarray = medial_points[1:]
        num_cylinders: int = len(start)
        axis: np.ndarray = end - start
        length: np.ndarray = np.linalg.norm(axis, axis=1)
        start -= global_shift
        radius: np.ndarray = branch.radii[1:]
        UnmodRadius: np.ndarray = branch.radii[:-1] # This field is occupied to store the bottom radius of the truncated cone
        axis /= length[:, np.newaxis]

        PositionInBranch: np.ndarray = np.arange(num_cylinders) + 1 # starts from 1 in TreeQSM
        BranchOrder: np.ndarray = np.full(num_cylinders, branch.order)
        added: np.ndarray = np.zeros(num_cylinders)
        added[:branch.active_medial_point_start_id] = 1
        branch_: np.ndarray = np.full(num_cylinders, branch_id)
        parent: np.ndarray = added_cylinder_count + PositionInBranch - 1
        if branch.parent_id != -1:
            parent[0] = branch_id_to_start_cylinder_id[branch.parent_id] +branch.joint_point_id
        else:
            parent[0] = 0
        extension: np.ndarray = added_cylinder_count + PositionInBranch + 1
        extension[-1] = 0

        qsm_cylinder_dict["radius"].append(radius.astype(np.float32))
        qsm_cylinder_dict["length"].append(length.astype(np.float32))
        qsm_cylinder_dict["start"].append(start.astype(np.float64))
        qsm_cylinder_dict["axis"].append(axis.astype(np.float32))
        qsm_cylinder_dict["parent"].append(parent.astype(np.uint64))
        qsm_cylinder_dict["extension"].append(extension.astype(np.uint64))
        qsm_cylinder_dict["added"].append(added.astype(np.bool_))
        qsm_cylinder_dict["UnmodRadius"].append(UnmodRadius.astype(np.float32))
        qsm_cylinder_dict["branch"].append(branch_.astype(np.uint64))
        qsm_cylinder_dict["BranchOrder"].append(BranchOrder.astype(np.uint64))
        qsm_cylinder_dict["PositionInBranch"].append(PositionInBranch.astype(np.uint64))

        branch_id_to_start_cylinder_id[branch_id] = added_cylinder_count
        added_cylinder_count += num_cylinders
    qsm_cylinder_dict = {k: np.concatenate(v, axis=0) for k, v in qsm_cylinder_dict.items()}
    
    # Extract parameters
    parameter_extraction: ParameterExtraction = ParameterExtraction(branch_id_to_branch, points, global_shift, **params_for_parameter_extraction)
    tree_data: Dict[int, Any] = {}
    for col_name, col_series in parameter_extraction.tree_dataframe.items():
        tree_data[col_name] = col_series.iloc[0]

    max_branch_id: int = max(branch_id_to_branch.keys())
    branch_data: Dict[str, np.ndarray] = {}
    for col_name, col_series in parameter_extraction.branch_dataframe.items():
        col_data: np.ndarray = np.zeros(shape=max_branch_id, dtype=col_series.dtype)
        branch_data[col_name] = col_data
    for _, row in parameter_extraction.branch_dataframe.iterrows():
        for col_name in parameter_extraction.branch_dataframe.columns:
            branch_data[col_name][row["id"] - 1] = row[col_name]
    branch_data["start"] = np.full(shape=max_branch_id, fill_value=np.iinfo(np.uint64).max, dtype=np.uint64)
    branch_data["end"] = np.full(shape=max_branch_id, fill_value=np.iinfo(np.uint64).max, dtype=np.uint64)
    for branch_id, triangle_id_range in branch_id_to_triangle_id_range.items():
        branch_data["start"][branch_id - 1] = triangle_id_range[0]
        branch_data["end"][branch_id - 1] = triangle_id_range[1]
    branch_data["id"][0] = 1 # Trunk

    if parameter_extraction.crown_convex_hull is not None:
        crown_convex_hull: o3d.geometry.TriangleMesh = parameter_extraction.crown_convex_hull
        crown_convex_hull.translate(-global_shift)
        crown_convex_hull.compute_triangle_normals()
        crown_convex_hull.compute_vertex_normals()
        o3d.io.write_triangle_mesh(crown_mesh_path, crown_convex_hull, write_vertex_normals=True)

    if parameter_extraction.active_crown_convex_hull is not None:
        active_crown_convex_hull: o3d.geometry.TriangleMesh = parameter_extraction.active_crown_convex_hull
        active_crown_convex_hull.translate(-global_shift)
        active_crown_convex_hull.compute_triangle_normals()
        active_crown_convex_hull.compute_vertex_normals()
        o3d.io.write_triangle_mesh(active_crown_mesh_path, active_crown_convex_hull, write_vertex_normals=True)
    
    # Change (N,) to (N, 1)
    for key in qsm_cylinder_dict.keys():
        if qsm_cylinder_dict[key].ndim == 1:
            qsm_cylinder_dict[key] = qsm_cylinder_dict[key].reshape(-1, 1)
    for key in branch_data.keys():
        if branch_data[key].ndim == 1:
            branch_data[key] = branch_data[key].reshape(-1, 1)
    
    qsm: Dict[str, Any] = {
        "QSM": {
            "cylinder": qsm_cylinder_dict,
            "branch": branch_data,
            "treedata": tree_data,
            "rundata": {
                "target_ply": os.path.basename(branch_mesh_path),
                "hash": branch_mesh_hash, # Prevent any tampering
                "projection": projection,
                "creator": creator,
                "qsmx_version": 1
            }
        }
    }

    savemat(qsm_path, qsm)
    return