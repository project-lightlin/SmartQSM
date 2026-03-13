import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from torch.nn.functional import pairwise_distance

class DisplacementTracker:
    _direction_losses: List[float]
    _distance_losses: List[float]
    _sample_counts: List[int]
    _errors: List[float]
    _use_log_distance: bool
    _eps: float

    def __init__(self, use_log_distance: bool= False, eps: float= 1e-8) -> None:
        self._direction_losses = []
        self._distance_losses = []
        self._sample_counts = []
        self._errors = []
        self._use_log_distance = use_log_distance
        self._eps = eps
        return
    

    def track(
            self, 
            *,
            pred_directions: torch.Tensor, 
            gt_directions: torch.Tensor,
            pred_distances: torch.Tensor,
            gt_distances: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        assert pred_directions.shape[0] == gt_directions.shape[0] == pred_distances.shape[0] == gt_distances.shape[0], "Input tensors must have the same size of the 1st dimension."
    
        if mask is not None:
            pred_directions = pred_directions[mask]
            gt_directions = gt_directions[mask]
            pred_distances = pred_distances[mask]
            gt_distances = gt_distances[mask]
        
        direction_loss: torch.Tensor = 1 - nn.CosineSimilarity().forward(pred_directions, gt_directions).mean()
        distance_loss: torch.Tensor
        if self._use_log_distance:
            distance_loss = nn.L1Loss().forward(torch.log(torch.clamp(pred_distances, min=self._eps)), torch.log(torch.clamp(gt_distances, min=self._eps)))
        else:
            distance_loss = nn.L1Loss().forward(pred_distances, gt_distances) 

        self._direction_losses.append(direction_loss.item())
        self._distance_losses.append(distance_loss.item())
        self._sample_counts.append(pred_directions.shape[0])
        self._errors.append(pairwise_distance(pred_distances, gt_distances, p=2).mean().item())

        return {"direction_loss": direction_loss, "distance_loss": distance_loss, "loss": direction_loss + distance_loss}
    
    def clear(self):
        self._direction_losses.clear()
        self._distance_losses.clear()
        self._sample_counts.clear()
        self._errors.clear()
        return

    def get_metrics(self) -> Dict[str, float]:
        direction_losses: np.ndarray = np.array(self._direction_losses)
        distance_losses: np.ndarray = np.array(self._distance_losses)
        sample_counts: np.ndarray = np.array(self._sample_counts)

        direction_loss: float = np.sum(direction_losses * sample_counts) / np.sum(sample_counts)
        distance_loss: float = np.sum(distance_losses * sample_counts) / np.sum(sample_counts)

        loss: float = direction_loss + distance_loss
        errors: np.ndarray = np.array(self._errors)
        error: float = np.sum(errors * sample_counts) / np.sum(sample_counts)

        return {
            "direction_loss": direction_loss,
            "distance_loss": distance_loss,
            "loss": loss,
            "error": error
        }
