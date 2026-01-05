
from scipy.spatial.distance import cdist
from scipy.optimize import Bounds
from utils.scipy_extra import wrapped_minimize, compute_aabb_bounds
import open3d as o3d
from typing import Callable, Dict, Any, Optional
import numpy as np
import open3d as o3d
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

def _l1_median_cost_fn(points: np.ndarray, center: np.ndarray) -> float:
    d: np.ndarray = np.linalg.norm(center - points, axis=1)
    return d.sum()

def _weiszfeld(
    X: np.ndarray,
    w: np.ndarray = None,
    x0: np.ndarray = None,
    max_iter: int = 1000,
    tol_abs: float = 1e-8,
    tol_rel: float = 1e-7,
    bounded: bool = False,
    lower_bounds: np.ndarray = None,
    upper_bounds: np.ndarray = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        assert w.shape == (n,), "w shape must be (n,)"
        # 允许非负权
        assert np.all(w >= 0), "weights must be non-negative"
    ws = w.sum()
    if ws == 0:
        raise ValueError("All weights are zero.")
    w = w / ws

    if x0 is None:
        x = np.average(X, axis=0, weights=w)
    else:
        x = np.asarray(x0, dtype=float).copy()
        assert x.shape == (d,)

    if bounded:
        assert lower_bounds is not None and upper_bounds is not None, "bounds required"
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        upper_bounds = np.asarray(upper_bounds, dtype=float)
        assert lower_bounds.shape == (d,) and upper_bounds.shape == (d,)
        x = np.minimum(np.maximum(x, lower_bounds), upper_bounds)

    for _ in range(max_iter):
        diffs = X - x  # (n, d)
        dist = np.linalg.norm(diffs, axis=1)  # (n,)
        zero_mask = dist < epsilon
        if np.any(zero_mask):
            return x.copy()

        inv_dist = 1.0 / np.maximum(dist, epsilon)
        weights = w * inv_dist  # (n,)

        numerator = (weights[:, None] * X).sum(axis=0)  # (d,)
        denominator = weights.sum()  # scalar
        x_new = numerator / max(denominator, epsilon)
        if bounded:
            x_new = np.minimum(np.maximum(x_new, lower_bounds), upper_bounds)
        step = np.linalg.norm(x_new - x)
        if step < tol_abs or step / max(1.0, np.linalg.norm(x)) < tol_rel:
            return x_new
        x = x_new
    return x  


def _wrapped_weiszfeld(data, init_center: np.ndarray, bounds: Optional[Bounds] = None, **kwargs) -> np.ndarray:
    if bounds is not None:
        center: np.ndarray = _weiszfeld(X=data, x0=init_center, bounded=True, lower_bounds=bounds.lb, upper_bounds=bounds.ub, **kwargs)
    else:
        center: np.ndarray = _weiszfeld(X=data, x0=init_center, **kwargs)
    return center

class _CPCOptimization:
    _prev_mu: Optional[float]
    _mu_fn: Callable[[np.ndarray], float]
    _sigma_fn: Callable[[np.ndarray], float]

    def __init__(self, mu_fn: Callable[[np.ndarray], float] = np.mean, sigma_fn: Callable[[np.ndarray], float] = np.std) -> None:
        self._prev_mu = None
        self._mu_fn = mu_fn
        self._sigma_fn = sigma_fn
        return
    
    def cost_fn(self, X: np.ndarray, v: np.ndarray) -> float:
        d: np.ndarray = np.linalg.norm(v - X, axis=1)  
        mu: float = self._mu_fn(d)
        m: int = d.shape[0]
        sigma2: float = self._sigma_fn(d) ** 2
        lambda_: float
        if self._prev_mu is None:
            lambda_ = m ** 2 * mu
        else:
            lambda_ = m ** 2 * self._prev_mu
        cost: float = d.sum() + lambda_ * sigma2
        self._prev_mu = mu
        return cost 

class CentroidEstimator3D:
    _points: np.ndarray
    _name_to_fn: Dict[str, Callable[[Dict[str, Any]], str]] 

    def __init__(self, points: np.ndarray) -> None:
        self._points = points
        self._name_to_fn = {
            "mass_center": lambda **kwargs: self.mass_center,
            #"centroid": lambda **kwargs: self.centroid,
            "l1_median": lambda **kwargs: self.l1_median(**kwargs),
            "trimmed_lad": lambda **kwargs: self.trimmed_lad(**kwargs),
            "ransac": lambda **kwargs: self.ransac(**kwargs),
            "cpc": lambda **kwargs: self.cpc(**kwargs),
            
        }
        return

    @property
    def mass_center(self) -> np.ndarray:
        return np.mean(self._points, axis=0)
    
    def _check_and_return_init_center(self, init_center_type: str) -> np.ndarray:
        assert init_center_type in ["mass_center"], \
            f"init_center_type for CPC should be in ['mass_center'], but got {init_center_type}"
        return self._name_to_fn[init_center_type]()

    def cpc(self, init_center_type: str = "mass_center", bounded: bool = False, **kwargs) -> np.ndarray:
        cpc_optimization: _CPCOptimization = _CPCOptimization(np.mean, np.std)
        return wrapped_minimize(
            cpc_optimization.cost_fn, 
            self._points,
            self._check_and_return_init_center(init_center_type), 
            compute_aabb_bounds(self._points) if bounded else None, 
            **kwargs
        )
    
    def l1_median(self, use_weiszfeld: bool = False, init_center_type: str = "mass_center", bounded: bool = False, **kwargs) -> np.ndarray:
        if use_weiszfeld:
            return wrapped_minimize(
                _l1_median_cost_fn, 
                self._points,
                self._check_and_return_init_center(init_center_type),
                compute_aabb_bounds(self._points) if bounded else None, 
                **kwargs
            )
        else:
            return _wrapped_weiszfeld(
                self._points, 
                self._check_and_return_init_center(init_center_type), 
                bounded, 
                **kwargs
            )
    
    def trimmed_lad(self, init_center_type: str = "mass_center", bounded: bool = False, inlier_rate: float = 0.7, **kwargs) -> np.ndarray:
        assert inlier_rate >= 0.5 and inlier_rate <= 1., "inlier_rate must be in [0.5, 1]"
        num_inliers = int(np.ceil(len(self._points) * inlier_rate))
        def lts_loss(center, points):
            d: np.ndarray = np.linalg.norm(center - points, axis=1)
            return np.sum(np.sort(d)[:num_inliers])
        return wrapped_minimize(
            lts_loss, 
            self._points,
            self._check_and_return_init_center(init_center_type), 
            compute_aabb_bounds(self._points) if bounded else None, 
            **kwargs
        )

    def ransac(self, init_center_type: str = "mass_center", bounded: bool = True, use_weiszfeld: bool = False, refitted: bool = True, inlier_rate: float = 0.7, confidence_or_iteration_count: float = 0.99, min_sample_count: int = 1, **kwargs) -> np.ndarray:
        assert inlier_rate > 0. and inlier_rate <= 1., f"inlier_rate must be in (0, 1]"
        assert confidence_or_iteration_count > 0., f"confidence must be in (0, 1) or an integer"

        num_iterations: int = np.ceil(np.log(1 - confidence_or_iteration_count) / np.log(1 - inlier_rate ** min_sample_count)).astype(int) if confidence_or_iteration_count < 1 else int(confidence_or_iteration_count)
        num_samples: int = len(self._points)
        num_inliers: int = np.ceil(inlier_rate * num_samples).astype(int)

        bounds: Optional[Bounds] = Bounds(
            np.min(self._points, axis=0),
            np.max(self._points, axis=0)
        ) if bounded else None

        init_center: np.ndarray = self._check_and_return_init_center(init_center_type)

        best_center: np.ndarray = np.copy(init_center)
        best_loss: float = np.inf

        for _ in range(num_iterations):
            sample_point_ids: np.ndarray = np.random.choice(num_samples, num_inliers, replace=False)
            sample_points: np.ndarray = self._points[sample_point_ids]
            estimated_center: np.ndarray 
            if use_weiszfeld:
                estimated_center = _wrapped_weiszfeld(
                    sample_points,
                    init_center,
                    bounds,
                    **kwargs
                )
            else:
                estimated_center = wrapped_minimize(
                    _l1_median_cost_fn,
                    sample_points,
                    init_center,
                    bounds,
                    **kwargs
                )
            errors: np.ndarray = np.linalg.norm(estimated_center - self._points, axis=1)
            
            if refitted:
                inlier_point_ids: np.ndarray = np.argsort(errors)[:num_inliers]
                inlier_points: np.ndarray = self._points[inlier_point_ids]
                if use_weiszfeld:
                    estimated_center = _wrapped_weiszfeld(
                        inlier_points,
                        estimated_center,
                        bounds,
                        **kwargs
                    )
                else:
                    estimated_center = wrapped_minimize(
                        _l1_median_cost_fn,
                        inlier_points,
                        estimated_center,
                        bounds,
                        **kwargs
                    )
                errors: np.ndarray = np.linalg.norm(estimated_center - self._points, axis=1)

            loss: float = np.sort(errors)[:num_inliers].mean()
            if loss < best_loss:
                best_loss = loss
                best_center = estimated_center
        return best_center

    def estimate(self, center_type: str, **kwargs):
        return self._name_to_fn[center_type](**kwargs)