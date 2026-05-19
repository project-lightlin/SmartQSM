from .cylinder_fitting import fit
import numpy as np
from typing import Tuple, Set, Generator
from .numpy_extra import calculate_distances_from_points_to_line
from math import comb
from itertools import combinations

from datetime import datetime, timedelta

from .parallel import *

def _generate_combos(
        num_points: int,
        sample_size: int,
        max_num_iterations: int
) -> List[Tuple[int, ...]]:
    total_combos: int = comb(num_points, sample_size)

    if total_combos <= max_num_iterations:
        return list(combinations(range(num_points), sample_size))
    
    rng: Generator = np.random.default_rng()
    checked_combos: Set[Tuple[int, ...]] = set()
    while len(checked_combos) < max_num_iterations:
        random_indices = rng.choice(
            num_points,
            size=sample_size,
            replace=False
        )

        combo = tuple(sorted(int(i) for i in random_indices))
        checked_combos.add(combo)

    return list(checked_combos)

def _evaluate_cylinder(
        points: np.ndarray,
        sample_point_indices: Tuple[int, ...],
        threshold: float
):
    sample_points = points[list(sample_point_indices)]

    try:
        w, c, radius, _ = fit(sample_points)
    except Exception:
        return None

    w = np.asarray(w, dtype=float)
    c = np.asarray(c, dtype=float)
    radius = float(radius)

    distances_to_axis = calculate_distances_from_points_to_line(
        points,
        c,
        c + w
    )

    residuals = np.abs(distances_to_axis - radius)

    inlier_mask = residuals <= threshold
    inliers = np.where(inlier_mask)[0]

    num_inliers = len(inliers)

    if num_inliers == 0:
        return None

    mean_residual = float(np.mean(residuals[inliers]))

    return num_inliers, mean_residual, w, c, radius, inliers
    

def cylindrical_ransac(
        points: np.ndarray,
        threshold: float = 0.05,
        max_num_iterations: int = 5000,
        sample_size: int = 8, # recommended by GPT
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if sample_size < 5:
        raise ValueError(f"sample_size must be at least 5")

    if points.shape[0] < sample_size:
        raise ValueError(f"points must have at least {sample_size} points")
    
    point_indices_per_combo: List[Tuple[int, ...]] = _generate_combos(
        num_points=len(points),
        sample_size=sample_size,
        max_num_iterations=max_num_iterations
    )

    chunk_size: int = np.ceil(len(point_indices_per_combo) / os.cpu_count()).astype(int)
    results = parallelize(
        (
            delayed(_evaluate_cylinder)(
                points,
                point_indices,
                threshold
            )
            for point_indices in point_indices_per_combo
        ),
        chunk_size=chunk_size
    )

    best_inliers: np.ndarray = np.array([], dtype=np.int64)
    best_w: np.ndarray = np.array([], dtype=float)
    best_c: np.ndarray = np.array([], dtype=float)
    best_radius: float = 0.0
    best_mean_residual: float = np.inf

    for result in results:
        if result is None:
            continue

        num_inliers, mean_residual, w, c, radius, inliers = result

        is_better = False

        if num_inliers > len(best_inliers):
            is_better = True
        elif num_inliers == len(best_inliers) and mean_residual < best_mean_residual:
            is_better = True

        if is_better:
            best_inliers = inliers
            best_w = w
            best_c = c
            best_radius = radius
            best_mean_residual = mean_residual

    
    if len(best_inliers) >= sample_size:
        try:
            refined_w, refined_c, refined_radius, _ = fit(points[best_inliers])

            refined_w = np.asarray(refined_w, dtype=float)
            refined_c = np.asarray(refined_c, dtype=float)
            refined_radius = float(refined_radius)

            refined_distances = calculate_distances_from_points_to_line(
                points,
                refined_c,
                refined_c + refined_w
            )

            refined_residuals = np.abs(refined_distances - refined_radius)
            refined_inliers = np.where(refined_residuals <= threshold)[0]

            best_w = refined_w
            best_c = refined_c
            best_radius = refined_radius
            best_inliers = refined_inliers

        except Exception:
            pass
    
    return best_w, best_c, best_radius, best_inliers