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

from .skeletonization_base import SkeletonizationBase
import numpy as np
from .thinning_algorithms import registry, CoreAlgorithmBase
from typing import Dict, Any, Optional, Tuple, List
import open3d as o3d

class ThinningBasedSkeletonization(SkeletonizationBase):
    _core_algorithm: Optional[CoreAlgorithmBase]
    _voxel_size_for_downsampling: Optional[float]

    def _clear(self) -> None:
        super()._clear()
        self._core_algorithm = None
        self._voxel_size_for_downsampling = None
        return
    
    def __init__(
            self,
            *,
            verbose: bool = False,
    ):
        super().__init__(
            verbose=verbose
        )

        self._clear()
        return

    def set_params(
            self, 
            *, 
            core_algorithm: str, 
            params_for_core_algorithm: Dict[str, Any] = {},
            voxel_size_for_downsampling: float = 0.01,
            edge_search_param: float = 0., 
            edge_weight_function = "squared_l2_norm", 
            topology_extractor = "spt"
    ):
        self._core_algorithm = registry[core_algorithm](
            verbose=self._verbose,
            **params_for_core_algorithm,
        )
        self._voxel_size_for_downsampling = voxel_size_for_downsampling
        
        super().set_params(
            edge_search_param=edge_search_param, 
            edge_weight_function=edge_weight_function, 
            topology_extractor=topology_extractor
        )
        super()._add_fns_to_pipeline(0, self._core_algorithm.get_pipeline() + [
            self._produce_skeletal_points_and_radii
        ])
        return
    
    def _convert_edge_search_param(self):
        return super()._convert_edge_search_param()
    
    def _produce_skeletal_points_and_radii(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.PointCloud]]:
        contracted_points: np.ndarray
        displacements: np.ndarray
        contracted_points, displacements = self._core_algorithm.output()

        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(contracted_points)
        point_ids_per_voxel: list[o3d.utility.IntVector]
        cloud, _, point_ids_per_voxel = cloud.voxel_down_sample_and_trace(voxel_size=self._voxel_size_for_downsampling, min_bound=cloud.get_min_bound(), max_bound=cloud.get_max_bound())
        sample_ids: List[int] = [point_ids[0] for point_ids in point_ids_per_voxel]
        self._skeletal_points = contracted_points[sample_ids]
        self._radii = np.linalg.norm(displacements[sample_ids], axis=1)

        if not self._verbose:
            return
        
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])

        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        for point_id, point in enumerate(self._skeletal_points):
            radius: float = self._radii[point_id]
            if radius <= 0.:
                continue
            sphere_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
            sphere_mesh.translate(point)
            sphere_mesh.compute_vertex_normals()
            mesh += sphere_mesh
        
        return f"Produced {len(self._skeletal_points)} skeletal points and estimated initial radii.", cloud, mesh

    def run(self, points: np.ndarray):
        self._core_algorithm.set_points(points=points)
        return super().run(points=points)