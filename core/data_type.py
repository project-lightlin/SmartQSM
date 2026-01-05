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

from typing import List, Optional
import open3d as o3d
import numpy as np

class Branch:
    parent_id: Optional[int] = None
    joint_point_id: Optional[int] = None
    order: Optional[int] = None
    medial_points: Optional[np.ndarray] = None
    active_medial_point_start_id: int = 0
    radii: Optional[np.ndarray] = None
    arterial_snake: Optional[o3d.geometry.TriangleMesh] = None # Used to store simplified output, including the connection section
    backup_arterial_snake: Optional[o3d.geometry.TriangleMesh] = None # Used for parameter extraction, removed the connection part and calculated as wood 
    base_radius: Optional[float] = None
    num_sectional_vertices: Optional[int] = None
