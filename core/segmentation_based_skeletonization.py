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
from typing import List, Union, Dict, Tuple, List, Any, Optional, Callable
import open3d as o3d
from joblib import delayed
import os
from utils.parallel import parallelize
from utils.centroid_estimator_3d import CentroidEstimator3D
from .skeletonization_base import SkeletonizationBase
from utils.open3d_extra import calculate_min_spacing_between
from .segmentation_algorithms import CoreAlgorithmBase, registry
from utils.cylinder_fitting import fit_cylinder
from utils.sklearn_extra import calculate_best_gaussian_mixture
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import warnings

class _CentroidEstimator:
    _centroid_estimator: Callable[[np.ndarray], Tuple[np.ndarray, float]]
    
    def __init__(self, solver: str, **kwargs) -> None:
        if solver == "cylinder_fitting":
            self._centroid_estimator = lambda points, kwargs=kwargs: fit_cylinder(points, **kwargs)[1:]
        else:
            self._centroid_estimator = lambda points, solver=solver, kwargs=kwargs: (
                CentroidEstimator3D(points).estimate(center_type=solver, **kwargs), 
                None
            )
        return
    
    def estimate(self, points: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        centroid: np.ndarray
        radius: Optional[float]
        try:
            centroid, radius = self._centroid_estimator(points)
        except (RuntimeError, ValueError):
            centroid = np.mean(points, axis=0)
            radius = None
        return centroid, radius

class _RadiusEstimator:
    _radius_estimator: Callable[[np.ndarray, np.ndarray], float]

    @staticmethod
    def _obtain_mean_and_stddev_of_max_component(gmm: Union[GaussianMixture, BayesianGaussianMixture]) -> Tuple[float, float]:
        means: np.ndarray = gmm.means_
        covariances: np.ndarray = gmm.covariances_
        weights: np.ndarray = gmm.weights_
        max_weight_id: int = np.argmax(weights)
        return means[max_weight_id], np.sqrt(covariances[max_weight_id])

    def __init__(self, solver: str ="gmm", **kwargs) -> None:
        if solver == "gmm":
            self._radius_estimator = lambda points, centroid: self._obtain_mean_and_stddev_of_max_component(
                calculate_best_gaussian_mixture(
                    np.linalg.norm(points - centroid, axis=1).reshape(-1, 1), 
                    **kwargs
                )
            )[0].item()
        elif solver =="cylinder_fitting":
            self._radius_estimator = lambda points, centroid: fit_cylinder(points, **kwargs)[2]
        else:
            raise NotImplementedError(f"Radius estimator solver {solver} is not implemented.")
        return
    
    def estimate(self, points: np.ndarray, centroid: np.ndarray, radius: Optional[float]) -> float:
        if radius is not None:
            return radius
        if len(points) == 1:
            return np.linalg.norm(points[0] - centroid)
        elif len(points) == 0:
            return 0.
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                radius = self._radius_estimator(points, centroid)
        except AttributeError:
            radius = 0.
        return radius

class SegmentationBasedSkeletonization(SkeletonizationBase):
    _core_algorithm: Optional[CoreAlgorithmBase]
    _edge_weight_function: Optional[str]
    _skeleton_extractor: Optional[str]
    _centroid_estimator: Optional[_CentroidEstimator]
    _radius_estimator: Optional[_RadiusEstimator]
    _temp_radii: Optional[List[float]]

    def _clear(self) -> None:
        super()._clear()
        
        self._edge_weight_function = None
        self._centroid_estimator = None
        self._radius_estimator = None
        self._core_algorithm = None
        self._temp_radii = None
        return
    
    def __init__(
        self, 
        *,
        verbose: bool = False
    ) -> None:
        super().__init__(
            verbose=verbose
        )

        self._add_edge_weight_function(
            "min_spacing", 
            lambda u_data, v_data: calculate_min_spacing_between(self._points[u_data[1]], self._points[v_data[1]])
        )
        return
    
    def set_params(
            self, 
            *,
            core_algorithm: str,
            params_for_core_algorithm: Dict[str, Any] = {},
            centroid_estimator: str = "mass_center", 
            params_for_centroid_estimator: Dict[str, Any] = {}, 
            radius_estimator: str = "gmm",
            params_for_radius_estimator: Dict[str, Any] = {}, 
            edge_search_param: float = 0.,
            edge_weight_function: str = "squared_l2_norm",
            topology_extractor: str = "spt",
    ) -> None:
        self._core_algorithm = registry[core_algorithm](
            verbose=self._verbose,
            **params_for_core_algorithm
        )

        self._edge_weight_function = edge_weight_function
        self._centroid_estimator = _CentroidEstimator(solver=centroid_estimator, **params_for_centroid_estimator)
        self._radius_estimator = _RadiusEstimator(solver=radius_estimator, **params_for_radius_estimator)
        
        super().set_params(
            edge_search_param=edge_search_param,
            edge_weight_function=edge_weight_function, 
            topology_extractor=topology_extractor
        )

        self._add_fns_to_pipeline(0, self._core_algorithm.get_pipeline() + [
            self._produce_skeletal_points
        ])

        self._add_fns_to_pipeline(len(self._pipeline), [
            self._produce_init_radii
        ])
        return
    
    def _produce_skeletal_points(self) -> Optional[Tuple[str, o3d.geometry.PointCloud]]:
        skeletal_points: List[np.ndarray]
        radii: List[float]

        self._clusters, self._edges = self._core_algorithm.output()
        
        result: Any = parallelize(
            (
                delayed(
                    self._centroid_estimator.estimate
                )(self._points[cluster]) 
                for cluster in self._clusters
            ),
            np.ceil(len(self._clusters) / os.cpu_count()).astype(int)
        )
        skeletal_points, radii = zip(*result)
        
        self._skeletal_points = np.array(skeletal_points)
        self._temp_radii = radii

        if not self._verbose:
            return
        
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self._skeletal_points)
        cloud.paint_uniform_color([1.0, 0.0, 0.0])
        
        return f"Produced {len(self._skeletal_points)} skeletal points.", cloud

    def _produce_init_radii(self) -> Optional[Tuple[str, o3d.geometry.TriangleMesh]]:
        result: Any = parallelize(
            (
                delayed(
                    self._radius_estimator.estimate
                )(self._points[cluster], self._skeletal_points[i], self._temp_radii[i]) 
                for i, cluster in enumerate(self._clusters)
            ),
            np.ceil(len(self._clusters) / os.cpu_count()).astype(int)
        )
        self._radii = np.array(result)
        
        if not self._verbose:
            return
        
        mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
        for point_id, point in enumerate(self._skeletal_points):
            radius: float = self._radii[point_id]
            if radius <= 0.:
                continue
            sphere_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
            sphere_mesh.translate(point)
            sphere_mesh.compute_vertex_normals()
            mesh += sphere_mesh
        return f"Estimated the initial radii (max={np.max(self._radii):.4f}).", mesh


    def _convert_edge_search_param(self):
        return super()._convert_edge_search_param()

    def run(self, points: np.ndarray):
        self._core_algorithm.set_points(points)
        return super().run(points=points)