import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

def create_cylinder(
        start_point: np.ndarray,
        end_point: np.ndarray,
        radius: float,
        resolution: int = 20
) -> o3d.geometry.TriangleMesh:
    z_axis: np.ndarray = np.array([0, 0, 1])
    height: float = np.linalg.norm(end_point - start_point)
    cylinder: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=radius,
                    height=height,
                    resolution=resolution
    )
    direction: np.ndarray = (end_point - start_point) / height
    axis = np.cross(z_axis, direction)
    norm_axis = np.linalg.norm(axis)

    if norm_axis < 1e-8:
        if np.dot(z_axis, direction) > 0:
            R = np.eye(3)
        else:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.pi * np.array([1, 0, 0]))
    else:
        axis /= norm_axis
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cylinder.rotate(R, center=np.zeros(3))
    cylinder.translate((start_point + end_point) / 2) 
    cylinder.compute_vertex_normals()
    return cylinder

def calculate_min_spacing_between(point_set_1: np.ndarray, point_set_2: np.ndarray) -> float:
    min_spacing: float
    if len(point_set_1) < len(point_set_2):
        kdtree = KDTree(point_set_2)
        min_spacing = np.min(kdtree.query(point_set_1, k=1)[0])
    else:
        kdtree = KDTree(point_set_1)
        min_spacing = np.min(kdtree.query(point_set_2, k=1)[0]) 
    return min_spacing