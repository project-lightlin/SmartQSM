
import open3d as o3d
import numpy as np
from typing import List, Dict, Any, Callable, Optional

class Clustering:
    _clustering_algorithm: Callable
    _global_kwargs: Dict[str, Any]

    def __init__(self, clustering_algorithm: str, **kwargs: Dict[str, Any]) -> None:
        registry = {
            "dbscan": Clustering.dbscan
        }
        self._clustering_algorithm = registry[clustering_algorithm]
        self._global_kwargs = kwargs
        return
    
    def __call__(self, points: np.ndarray, **kwargs) -> List[np.ndarray]:
        local_kwargs = self._global_kwargs.copy()
        local_kwargs.update(kwargs)
        labels: np.ndarray = np.array(self._clustering_algorithm(points, **local_kwargs))
        return [np.where(labels == label)[0] for label in np.unique(labels) if label != -1]

    @staticmethod
    def dbscan(points: np.ndarray, eps: float = 0.04, min_points: int = 5):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points))
    