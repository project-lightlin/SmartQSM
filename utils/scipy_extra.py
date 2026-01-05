from scipy.optimize import minimize, Bounds, OptimizeResult
import numpy as np
from typing import Callable, Optional, List, Dict, Any

def wrapped_minimize(
        costf: Callable[[np.ndarray, np.ndarray], float], 
        data: np.ndarray, 
        x0: np.ndarray, 
        bounds: Optional[Bounds] = None, 
        constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs
) -> np.ndarray:
    x: np.ndarray
    result: OptimizeResult = minimize(
        lambda x: costf(data, x), 
        x0, 
        bounds=bounds,
        constraints=constraints,
        **kwargs
    )
    x = result.x
    return x

def compute_aabb_bounds(points: np.ndarray) -> Bounds:
    lower_bounds: np.ndarray = np.min(points, axis=0)
    upper_bounds: np.ndarray = np.max(points, axis=0)
    return Bounds(lower_bounds, upper_bounds)