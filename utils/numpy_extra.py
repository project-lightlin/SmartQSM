import numpy as np
from sklearn.decomposition import PCA

def calculate_three_point_curvature(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    # k = 4S(Triangle ABC)/(abc) = 2 * |(p2-p1) × (p3-p2)| / (|p2-p1| * |p3-p2| * |p3-p1|)
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    cross = np.linalg.norm(np.cross(p2-p1, p3-p2))
    abc = a * b * c
    if abc == 0:
        return 0 
    curvature = 2 * cross / abc
    return curvature

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
    cos_angle = dot_product / norm_product
    angle = np.degrees(np.arccos(cos_angle))
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

def calculate_distances_between_points_and_surface(Ps: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
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

def get_projection_vector(plane_normal_vector: np.ndarray, vector_from_a_plane_point: np.ndarray) -> np.ndarray:
    if len(plane_normal_vector) != 3:
        raise ValueError("plane_coefficients must be [A, B, C] in equation Ax+By+Cz+D=0.")
    
    n : np.ndarray = plane_normal_vector
    v : np.ndarray = vector_from_a_plane_point
    return v - np.dot(n, v) / (np.linalg.norm(n) ** 2) * n

def sample_line_segment(p1, p2, max_step=0.001):

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    
    diffs = np.abs(p2 - p1)
    max_delta = np.max(diffs)
    
    n_points = np.ceil(max_delta / max_step).astype(int) + 1
    t_values = np.linspace(0, 1, n_points)
    
    points = p1 + np.outer(t_values,  p2 - p1)
    return points

def point_to_segment_distances_3d(point: np.ndarray, segment_starts: np.ndarray, segment_ends: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    K = np.asarray(point, dtype=float).reshape(1, 3)
    M = np.asarray(segment_starts, dtype=float)
    Np = np.asarray(segment_ends, dtype=float)
    v = Np - M                     
    w = K - M                      
    vv = np.einsum('ij,ij->i', v, v)         
    vw = np.einsum('ij,ij->i', v, w)        
    mask_deg = vv < eps
    # t = clamp(vw / vv, 0, 1)
    t = np.zeros_like(vw)
    t[~mask_deg] = vw[~mask_deg] / vv[~mask_deg]
    t = np.clip(t, 0.0, 1.0)
    P = M + t[:, None] * v         # (N,3)
    P[mask_deg] = M[mask_deg]
    d = np.linalg.norm(P - K, axis=1)
    return d, P, t

def calculate_heading_angle(heading_direction: np.ndarray, reference_direction: np.ndarray = np.array([0., 1.]), clockwise=True) -> float: # The convention for navigation is clockwise
    assert heading_direction.shape == reference_direction.shape, "heading_direction and reference_direction must have the same shape."
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