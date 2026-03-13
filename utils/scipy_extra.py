from scipy.optimize import minimize, Bounds, OptimizeResult
import numpy as np
from typing import Callable, Any
import warnings
from scipy.interpolate import BSpline
import cvxpy as cp

def wrapped_minimize(
    fun: Callable[..., float], 
    x0: Any, 
    **kwargs
) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res: OptimizeResult = minimize(
            fun, 
            x0, 
            **kwargs
        )
        if (not res.success) or (not np.all(np.isfinite(res.x))):
            return x0.copy()
        return res.x

def compute_aabb_bounds(points: np.ndarray) -> Bounds:
    lower_bounds: np.ndarray = np.min(points, axis=0)
    upper_bounds: np.ndarray = np.max(points, axis=0)
    return Bounds(lower_bounds, upper_bounds)

def berhu(delta, r, eps=1e-12):
    """
    Reverse Huber / BerHu:
      |r|                  , |r| <= delta
      (r^2 + delta^2)/(2d) , |r| >  delta
    """
    r = np.asarray(r)
    d = max(float(delta), eps)
    a = np.abs(r)
    return np.where(a <= d, a, (r*r + d*d) / (2.0*d))


def _as_1d(a, name):
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a

def _check_xyw(x, y, w=None):
    x = _as_1d(x, "x")
    y = _as_1d(y, "y")
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x/y must be finite")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    dx = np.diff(x)
    if np.any(dx <= 0):
        if w is None:
            w0 = np.ones_like(x)
        else:
            w0 = _as_1d(w, "w")[order]
        xu, inv = np.unique(x, return_inverse=True)
        yw = np.zeros_like(xu)
        ww = np.zeros_like(xu)
        for i in range(x.size):
            j = inv[i]
            yw[j] += y[i] * w0[i]
            ww[j] += w0[i]
        y = yw / np.maximum(ww, 1e-12)
        x = xu
        w = ww
    else:
        if w is not None:
            w = _as_1d(w, "w")[order]
    if w is None:
        w = np.ones_like(x)
    if np.any(w < 0) or np.any(~np.isfinite(w)):
        raise ValueError("w must be finite and nonnegative")
    return x, y, w

def _make_open_uniform_knots(x, n_basis, k):
    # open-uniform/clamped knots on [min(x), max(x)]
    a, b = float(np.min(x)), float(np.max(x))
    if n_basis <= k:
        raise ValueError("n_basis must be > k")
    n_internal = n_basis - (k + 1)
    if n_internal < 0:
        n_internal = 0
    if n_internal == 0:
        internal = np.array([], float)
    else:
        internal = np.linspace(a, b, n_internal + 2)[1:-1]
    t = np.r_[np.full(k+1, a), internal, np.full(k+1, b)]
    return t

def _bspline_design_matrix(x, t, k):
    x = np.asarray(x, float)
    n_basis = len(t) - k - 1
    coeffs = np.eye(n_basis)
    B = np.empty((x.size, n_basis), float)
    for j in range(n_basis):
        spl = BSpline(t, coeffs[j], k, extrapolate=False)
        B[:, j] = spl(x)
    B[~np.isfinite(B)] = 0.0
    return B

def _ispline_design_matrix(x, t, k, grid_size=2000):
    x = np.asarray(x, float)
    a, b = float(t[k]), float(t[-k-1])  
    grid = np.linspace(a, b, grid_size)
    B_grid = _bspline_design_matrix(grid, t, k)  # (G, m)

    dx = grid[1] - grid[0]
    I_grid = np.cumsum((B_grid[:-1] + B_grid[1:]) * 0.5 * dx, axis=0)
    I_grid = np.vstack([np.zeros((1, I_grid.shape[1])), I_grid])  # (G, m)

    denom = I_grid[-1, :]
    denom = np.where(denom <= 1e-14, 1.0, denom)
    I_grid = I_grid / denom

    A_I = np.empty((x.size, I_grid.shape[1]), float)
    for j in range(I_grid.shape[1]):
        A_I[:, j] = np.interp(x, grid, I_grid[:, j], left=0.0, right=1.0)
    return A_I

class MonotoneISplineQP:
    def __init__(self, x, y, w=None, t=None, k=3, n_basis=12,
                 lam=0.0, grid_size=2000, extrapolate="clamp"):
        x, y, w = _check_xyw(x, y, w)

        if t is None:
            t = _make_open_uniform_knots(x, n_basis=n_basis, k=k)
        else:
            t = np.asarray(t, float)

        self.x = x
        self.y = y
        self.w = w
        self.t = t
        self.k = int(k)
        self.extrapolate = extrapolate

        A_I = _ispline_design_matrix(x, t, k, grid_size=grid_size)  # (n, m)
        n, m = A_I.shape

        Aw = np.hstack([np.ones((n, 1)), A_I]) * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)

        if lam > 0:
            D = np.zeros((m-2, m))
            for i in range(m-2):
                D[i, i] = 1
                D[i, i+1] = -2
                D[i, i+2] = 1
            Dw = np.hstack([np.zeros((D.shape[0], 1)), D])  
        else:
            Dw = None

        beta = cp.Variable(m + 1)  # [a0, c...]

        obj = cp.sum_squares(Aw @ beta - yw)
        if lam > 0:
            obj += lam * cp.sum_squares(Dw @ beta)

        constraints = [beta[1:] >= 0]  # c >= 0
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP)

        if beta.value is None:
            raise RuntimeError("QP failed to solve (beta is None). Try different knots/lam/solver.")

        self.coef_ = np.array(beta.value).ravel()  # length m+1
        self._m = m

    def _A_I_pred(self, x_pred):
        x_pred = np.asarray(x_pred, float)
        a, b = float(self.t[self.k]), float(self.t[-self.k-1])
        if self.extrapolate == "clamp":
            xp = np.clip(x_pred, a, b)
        elif self.extrapolate is False or self.extrapolate is None:
            xp = x_pred
        else:
            xp = x_pred

        A_I = _ispline_design_matrix(xp.ravel(), self.t, self.k, grid_size=2000)
        return A_I.reshape(x_pred.size, -1)

    def __call__(self, x_pred):
        x_pred = np.asarray(x_pred, float)
        A_I = self._A_I_pred(x_pred).reshape(x_pred.size, -1)
        yhat = self.coef_[0] + A_I @ self.coef_[1:]
        return yhat.reshape(x_pred.shape)
    
