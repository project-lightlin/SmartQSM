import numpy as np
import open3d as o3d
from typing import List
from .numpy_extra import find_a_vertical_direction_3d, get_projection_vector
from scipy.spatial.transform import Rotation

def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def _project_and_normalize(v: np.ndarray, onto_perp_to: np.ndarray) -> np.ndarray:
    u = onto_perp_to
    v_proj = v - np.dot(v, u) * u
    n = np.linalg.norm(v_proj)
    return v_proj / n

# Vectorized Implementation by GPT-5
def generate_arterial_snake(
    points: np.ndarray,
    radii: np.ndarray,
    num_vertices: int
) -> o3d.geometry.TriangleMesh:
    if num_vertices < 3:
        raise ValueError("num_vertices must be >= 3")
    N = len(points)
    if N < 2:
        raise ValueError("at least 2 points.")
    points = np.asarray(points, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if len(radii) != N:
        raise ValueError("radii length must match points length.")

    start = np.empty_like(points)
    end   = np.empty_like(points)
    start[:-1] = points[:-1]
    end[:-1]   = points[1:]
    start[-1]  = points[-2]
    end[-1]    = points[-1]
    directions = end - start
    directions = _safe_normalize(directions)  # (N, 3)

    principal_normals = np.zeros_like(directions)
    principal_normals[0] = find_a_vertical_direction_3d(directions[0])

    for i in range(1, N - 0):  
        if i < N - 1:
            principal_normals[i] = _project_and_normalize(principal_normals[i - 1], directions[i])
        else:
            principal_normals[i] = principal_normals[i - 1]

    angles_deg = (np.arange(num_vertices) / num_vertices) * 360.0
    angles_rad = np.deg2rad(angles_deg)  # (M,)

    rotvec = directions[:, None, :] * angles_rad[None, :, None]  # (N, M, 3)

    rotvec_flat = rotvec.reshape(-1, 3)  # (N*M, 3)
    R_obj = Rotation.from_rotvec(rotvec_flat)

    base_vec = np.broadcast_to(principal_normals[:, None, :], rotvec.shape).reshape(-1, 3)  # (N*M, 3)
    rotated = R_obj.apply(base_vec).reshape(N, num_vertices, 3)  # (N, M, 3)

    centers = np.copy(points)
    centers[:-1] = points[:-1]       # i < N-1 -> start_point=points[i]
    centers[-1]  = points[-1]        # i == N-1 -> end_point=points[-1]
    centers = centers[:, None, :]    # (N,1,3) 

    radii_b = radii[:, None, None]   # (N,1,1)

    vertices = centers + rotated * radii_b  # (N, M, 3)
    vertices_flat = vertices.reshape(-1, 3)  # (N*M, 3)

    if N >= 2:
        i_idx = np.arange(1, N)[:, None]        # (N-1,1)
        j_idx = np.arange(num_vertices)[None, :] # (1,M)

        a = (i_idx * num_vertices + j_idx).reshape(-1)
        b = ((i_idx - 1) * num_vertices + j_idx).reshape(-1)
        c = ((i_idx - 1) * num_vertices + (j_idx + 1) % num_vertices).reshape(-1)
        d = (i_idx * num_vertices + (j_idx + 1) % num_vertices).reshape(-1)

        tri1 = np.stack([a, b, c], axis=-1)
        tri2 = np.stack([a, c, d], axis=-1)
        triangles = np.vstack([tri1, tri2]).astype(np.int32)
    else:
        triangles = np.empty((0, 3), dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_flat)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh.compute_triangle_normals()
    return mesh


def calculate_rough_volume(arterial_snake_mesh: o3d.geometry.TriangleMesh, num_vertices: int) -> float:
    vertices = np.asarray(arterial_snake_mesh.vertices, dtype=float)
    n_layers = len(vertices) // num_vertices

    L = n_layers - 1 
    K = num_vertices - 2 

    layer_start = np.arange(L) * num_vertices
    next_layer_start = layer_start + num_vertices

    p0 = vertices[layer_start]    
    q0 = vertices[next_layer_start] 

    idx_offset = 1 + np.arange(K)  
    idx_p1 = layer_start[:, None] + idx_offset[None, :]
    idx_p2 = idx_p1 + 1
    idx_q1 = next_layer_start[:, None] + idx_offset[None, :]
    idx_q2 = idx_q1 + 1

    p1_array = vertices[idx_p1] 
    p2_array = vertices[idx_p2]
    q1_array = vertices[idx_q1]
    q2_array = vertices[idx_q2]

    # V1
    p0q0 = q0[:, None, :] - p0[:, None, :]       
    p0p1 = p1_array - p0[:, None, :]
    p0p2 = p2_array - p0[:, None, :]
    cross1 = np.cross(p0q0, p0p1)
    V1 = np.abs(np.sum(cross1 * p0p2, axis=2)) / 6.0

    # V2
    q1p1 = p1_array - q1_array
    q1q0 = q0[:, None, :] - q1_array 
    q1p2 = p2_array - q1_array
    cross2 = np.cross(q1p1, q1q0)
    V2 = np.abs(np.sum(cross2 * q1p2, axis=2)) / 6.0

    # V3
    q2q1 = q1_array - q2_array
    q2q0 = q0[:, None, :] - q2_array
    q2p2 = p2_array - q2_array
    cross3 = np.cross(q2q1, q2q0)
    V3 = np.abs(np.sum(cross3 * q2p2, axis=2)) / 6.0

    volume = np.sum(V1 + V2 + V3)

    return volume
