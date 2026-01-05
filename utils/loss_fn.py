import numpy as np
from typing import Optional, Dict, Callable

def _reduce(loss: np.ndarray, reduction: str) -> float:
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise NotImplementedError(f"reduction method '{reduction}' not implemented")

def _weight(loss: np.ndarray, weights: Optional[np.ndarray] = None):
    if weights is None:
        return loss
    else:
        return weights * loss

def l1_loss(input: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None, reduction: str = "mean") -> float:
    loss = np.abs(input - target)
    return _reduce(_weight(loss, weights), reduction)


def l2_loss(input: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None, reduction: str = "mean") -> float:
    loss = (input - target) ** 2
    return _reduce(_weight(loss, weights), reduction)

def huber_loss(input: np.ndarray, target: np.ndarray, delta: float = 1.0, weights: Optional[np.ndarray] = None, reduction: str = "mean") -> float:
    r = input - target
    abs_r = np.abs(r)
    out = np.empty_like(abs_r)
    mask = abs_r <= delta
    out[mask] = 0.5 * (r[mask] ** 2)
    out[~mask] = delta * (abs_r[~mask] - 0.5 * delta)
    return _reduce(_weight(out, weights), reduction)

def tukey_loss(input: np.ndarray, target: np.ndarray, c: float = 4.685, weights: Optional[np.ndarray] = None, reduction: str = "mean") -> float:
    r = input - target
    mask = np.abs(r) <= c
    out = np.zeros_like(r, dtype=float)
    x = r[mask] / c
    out[mask] = (c**2 / 6) * (1 - (1 - x**2) ** 3)
    out[~mask] = c**2 / 6
    return _reduce(_weight(out, weights), reduction)

def soft_l1_loss(input: np.ndarray, target: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 reduction: str = "mean") -> float:
    r = input - target
    loss = 2 * (np.sqrt(1 + r**2) - 1)
    return _reduce(_weight(loss, weights), reduction)

def cauchy_loss(input: np.ndarray, target: np.ndarray,
                weights: Optional[np.ndarray] = None,
                reduction: str = "mean") -> float:
    r = input - target
    loss = np.log1p(r**2)
    return _reduce(_weight(loss, weights), reduction)

def arctan_loss(input: np.ndarray, target: np.ndarray,
                weights: Optional[np.ndarray] = None,
                reduction: str = "mean") -> float:
    r = input - target
    loss = np.arctan(r**2)
    return _reduce(_weight(loss, weights), reduction)

def cosine_loss(input: np.ndarray, target: np.ndarray,
                weights: Optional[np.ndarray] = None,
                reduction: str = "mean", eps = 1e-12) -> float:
    norm_input = np.linalg.norm(input, axis=-1)
    norm_target = np.linalg.norm(target, axis=-1)
    loss = 1 - np.sum(input * target, axis=-1) / (norm_input * norm_target + eps)
    return _reduce(_weight(loss, weights), reduction)

def offset_loss(input: np.ndarray, target: np.ndarray,
                weights: Optional[np.ndarray] = None, l1_weight: float = 1., cosine_weight: float = 1.,
                reduction: str = "mean", eps = 1e-12) -> float:
    in_offset_norm = np.linalg.norm(input, axis=-1) + eps
    in_offset_dir = input / in_offset_norm[:, None]
    target_offset_norm = np.linalg.norm(target, axis=-1) + eps
    target_offset_dir = target / target_offset_norm[:, None]
    loss = l1_weight * np.abs(in_offset_norm - target_offset_norm) + \
           cosine_weight * (1 - np.sum(in_offset_dir * target_offset_dir, axis=-1))
    return _reduce(_weight(loss, weights), reduction)


loss_name_to_fn: Dict[str, Callable] = {
    "l1": l1_loss,
    "l2": l2_loss,
    "huber": huber_loss,
    "tukey": tukey_loss,
    "soft_l1": soft_l1_loss,
    "cauchy": cauchy_loss,
    "arctan": arctan_loss,
    "cosine": cosine_loss,
    "offset": offset_loss
}