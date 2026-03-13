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

from .skeletonization_pipeline_base import SkeletonizationPipelineBase
import numpy as np
from .core_algorithms import *
from typing import Dict, Any, Optional, Tuple, List
import open3d as o3d

class ThinningBasedSkeletonizationPipeline(SkeletonizationPipelineBase):
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
            core_algorithm_kwargs: Dict[str, Any] = {},
            voxel_size_for_downsampling: float = 0.01,
            **kwargs
    ):
        self._core_algorithm = registry["thinning"][core_algorithm](
            verbose=self._verbose,
            **core_algorithm_kwargs,
        )
        self._voxel_size_for_downsampling = voxel_size_for_downsampling
        super().set_params(
            **kwargs
        )
        super()._add_fns_to_pipeline(0, self._core_algorithm.get_pipeline() + [
            self._produce_skeletal_points_and_radii
        ])
        return
    
    def _produce_skeletal_points_and_radii(self) -> Optional[Tuple[str, o3d.geometry.PointCloud, o3d.geometry.PointCloud]]:
        output: Dict[str, Any] = self._core_algorithm.output()
        contracted_points: np.ndarray = output["contracted_points"]
        displacements: np.ndarray = output["displacements"]

        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(contracted_points)
        point_indices_per_voxel: list[o3d.utility.IntVector]
        cloud, _, point_indices_per_voxel = cloud.voxel_down_sample_and_trace(voxel_size=self._voxel_size_for_downsampling, min_bound=cloud.get_min_bound(), max_bound=cloud.get_max_bound())
        sample_ids: List[int] = [point_indices[0] for point_indices in point_indices_per_voxel]
        self._skeletal_points = contracted_points[sample_ids]
        displacements = displacements[sample_ids]
        self._radii = np.linalg.norm(displacements, axis=1)

        if not self._verbose:
            return
        
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])

        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        for point_idx, point in enumerate(self._skeletal_points):
            radius: float = self._radii[point_idx]
            if radius <= 0.:
                continue
            sphere_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
            sphere_mesh.translate(point)
            sphere_mesh.compute_vertex_normals()
            mesh += sphere_mesh
        
        return f"Produced {len(self._skeletal_points)} skeletal points and estimated initial radii.", cloud, mesh

    def run(self, points: np.ndarray):
        self._core_algorithm.input(points=points)
        return super().run(points=points)