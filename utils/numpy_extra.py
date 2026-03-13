import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.maximum(v_norm, eps)
    return v / v_norm

def calculate_three_point_curvatures(P1s: np.ndarray, P2s: np.ndarray, P3s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v21 = P2s - P1s              # (..., 3)
    v32 = P3s - P2s              # (..., 3)
    v31 = P3s - P1s              # (..., 3)
    a = np.linalg.norm(v21, axis=-1)   # (...)
    b = np.linalg.norm(v32, axis=-1)   # (...)
    c = np.linalg.norm(v31, axis=-1)   # (...)
    cross = np.linalg.norm(np.cross(v21, v32), axis=-1)  # (...)
    abc = a * b * c                                     # (...)
    valid = abc > eps
    curvatures = np.zeros_like(abc, dtype=float)
    curvatures[valid] = 2.0 * cross[valid] / abc[valid]

    return curvatures

def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return np.nan
    cos_angle = dot_product / norm_product
    angle = np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def calculate_distances_from_points_to_line(Ps: np.ndarray, M: np.ndarray, N: np.ndarray) -> np.ndarray:
    Ps = np.array(Ps)
    M = np.array(M)
    N = np.array(N)
    
    MN = N - M
    
    denominator_scalar = np.dot(MN, MN)
    if denominator_scalar == 0:
        raise ValueError("M and N are the same point, cannot form a line.")
    
    MP_vectors = Ps - M
    
    proj_lengths = np.dot(MP_vectors, MN) / denominator_scalar
    
    proj_vectors = proj_lengths[:, np.newaxis] * MN
    
    h_vectors = MP_vectors - proj_vectors
    
    distances = np.linalg.norm(h_vectors, axis=1)
    
    return distances

def calculate_distances_from_points_to_surface(Ps: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)
    n_len = np.linalg.norm(n)
    if n_len == 0 or not np.isfinite(n_len):
        return np.full(Ps.shape[0], np.nan, dtype=float)

    APs = Ps - A
    signed_dist = (APs @ n) / n_len          # (N,)
    return np.abs(signed_dist).astype(float)

def find_a_vertical_direction_3d(direction: np.ndarray) -> np.ndarray:
    direction: np.ndarray = direction / np.linalg.norm(direction)
    x: float = direction[0]
    y: float = direction[1]
    z: float = direction[2]
    vertical_direction: np.ndarray
    if abs(y) >= abs(x) and abs(z) >= abs(x):
        vertical_direction = np.array([0., -z, y])
    elif abs(x) >= abs(y) and abs(z) >= abs(y):
        vertical_direction = np.array([-z, 0., x])
    else:
        vertical_direction = np.array([-y, x, 0.])
    return vertical_direction / np.linalg.norm(vertical_direction)

def project_onto_plane(n: np.ndarray, v: np.ndarray) -> np.ndarray:
    if len(v) != 3:
        raise ValueError("vector must be [x, y, z] in equation Ax+By+Cz+D=0.")
    
    n_norm2 = np.dot(n, n)
    if n_norm2 == 0:
        raise ValueError("Plane normal vector must be non-zero.")

    return v - np.dot(n, v) / n_norm2 * n

def sample_line_segment(p1, p2, max_step=0.001):

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    
    diffs = np.abs(p2 - p1)
    max_delta = np.max(diffs)
    
    n_points = np.ceil(max_delta / max_step).astype(int) + 1
    t_values = np.linspace(0, 1, n_points)
    
    points = p1 + np.outer(t_values,  p2 - p1)
    return points

def calculate_distances_from_points_to_segments_3d(
    points: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)          # (M,3)
    M = np.asarray(segment_starts, dtype=float).reshape(-1, 3)       # (N,3)
    Np = np.asarray(segment_ends, dtype=float).reshape(-1, 3)        # (N,3)

    K = points[:, None, :]   # (M,1,3)
    M_exp = M[None, :, :]    # (1,N,3)
    Np_exp = Np[None, :, :]  # (1,N,3)

    v = Np_exp - M_exp       # (1,N,3)
    w = K - M_exp            # (M,N,3)

    vv = np.einsum('ij,ij->i', M, Np - M) 

    vw = np.einsum('ijk,ijk->ij', w, v)    # (M,N,3)->(M,N)

    mask_deg = vv < eps                    # (N,)

    t = np.zeros_like(vw)                  # (M,N)
    vv_safe = vv.copy()
    vv_safe[mask_deg] = 1.0                # 避免除零
    t = vw / vv_safe[None, :]             # (M,N)
    t = np.clip(t, 0.0, 1.0)

    P = M_exp + t[..., None] * v          # (M,N,3)

    if np.any(mask_deg):
        P[:, mask_deg, :] = M_exp[:, mask_deg, :]

    d = np.linalg.norm(P - K, axis=2)     # (M,N)
    return d, P, t

def calculate_heading_angle(heading_direction: np.ndarray, reference_direction: np.ndarray = np.array([0., 1.]), clockwise=True) -> float: # The convention for navigation is clockwise
    assert heading_direction.shape == reference_direction.shape \
        and heading_direction.ndim == 1 \
        and heading_direction.size == 2, "heading_direction and reference_direction must have the same shape. Both directions must be 1D vectors of length 2, like np.array([0, 1])"
    
    if np.linalg.norm(heading_direction) == 0.:
        return 0.
    if np.linalg.norm(reference_direction) == 0.:
        raise ValueError("Reference_direction must be non-zero vectors.")
    x = reference_direction / np.linalg.norm(reference_direction)
    y = heading_direction / np.linalg.norm(heading_direction)
    cos_angle = np.dot(x, y)
    angle = np.arccos(cos_angle) / np.pi * 180 # np.arccos => 0~pi
    cross = x[0] * y[1] - x[1] * y[0]
    if not clockwise:
        if cross < 0.:
            angle = 360. - angle
    else:
        if cross > 0.:
            angle = 360. - angle
    if angle == 360.: # Due to the loss of numerical accuracy
        return 0.
    else:
        return angle
    
def calculate_direction_of_ordered_points(points: np.array) -> np.array:
    if points.shape[0] < 2:
        raise ValueError
    elif points.shape[0] == 2:
        direction = points[1,:]-points[0,:]
        return direction / np.linalg.norm(direction)
    pca = PCA(n_components=1)
    pca.fit(points)
    principal_component = pca.components_[0]
    principal_component = principal_component / np.linalg.norm(principal_component)
    distance = np.linalg.norm(points[-1] - points[0])
    if np.linalg.norm(points[0] + distance * principal_component - points[-1]) \
        < np.linalg.norm(points[0] - distance * principal_component - points[-1]):
        return principal_component
    else:
        return -principal_component
    
def calculate_symmetric_points(
    points: np.ndarray, 
    line_start: np.ndarray, 
    line_end: np.ndarray
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    line_start = np.asarray(line_start, dtype=np.float64).reshape(3)
    line_end = np.asarray(line_end, dtype=np.float64).reshape(3)
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be a (N, 3) array, but got {points.shape}")
    
    # 计算直线的方向向量
    v = line_end - line_start
    v_norm_sq = np.dot(v, v)
    
    if v_norm_sq < 1e-12:
        raise ValueError("line_start and line_end must be different points.")
    
    # 向量化计算
    AP = points - line_start
    
    t = np.dot(AP, v) / v_norm_sq
    
    Q = line_start + t[:, np.newaxis] * v
    
    symmetric_points = 2 * Q - points
    
    return symmetric_points