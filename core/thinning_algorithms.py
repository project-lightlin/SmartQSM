# SmartQSM - project-lightlin.github.io
# 
# Copyright (C) 2025-, YANG Jie <nj_yang_jie@foxmail.com>
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from .core_algorithm_base import CoreAlgorithmBase
import numpy as np
from typing import Optional, Tuple, List, Any, Dict
import open3d as o3d
from scipy.spatial import cKDTree
import os

class SparseConvBasedContraction(CoreAlgorithmBase):
    _epoch: int
    _contracted_points: Optional[np.ndarray]
    _displacements: Optional[np.ndarray]
    _max_iteration: int
    _cube_size: float
    _buffer_size: float
    _voxel_size: float
    _device: str
    _batch_size: int
    _model: Any
    _chamfer_distance: Optional[float]
    _early_stopped: bool

    def __init__(self, *, ckpt_path: str, batch_size: int, max_iteration: int, device: str = "cuda", voxel_size: float = 0.01, cube_size: float = 4.0, buffer_size: float = 0.4, verbose: bool = False, use_chamfer_distance: bool = False) -> None:
        super().__init__(verbose=verbose)
        self._verbose = verbose
        self._max_iteration = max_iteration
        self._epoch = 0

        self._cube_size = cube_size
        self._buffer_size = buffer_size
        self._voxel_size = voxel_size
        self._device = device
        self._batch_size = batch_size
        from .SmartTreeXX.models.smart_tree_xx import SmartTreeXX
        import torch
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "SmartTreeXX", ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        self._model = SmartTreeXX(
            **ckpt["hparams"]
        ).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        
        self._chamfer_distance = float("inf") if use_chamfer_distance else None
        self._early_stopped = False
        return
    
    def _contract(self) -> Tuple[str, o3d.geometry.PointCloud]:
        self._epoch += 1
        if self._early_stopped:
            if not self._verbose:
                return
            return (f"Skipped the {self._epoch}th contraction.", None)
        if self._epoch == 1:
            self._contracted_points = np.copy(self._points).astype(np.float32)
            self._displacements = np.zeros_like(self._points).astype(np.float32)
        
        from .SmartTreeXX.datasets.synthetic_tree import SingleTreeDataset, collate_synthetic_tree
        import torch
        dataset: SingleTreeDataset = SingleTreeDataset(
            np.concatenate((self._contracted_points, self._displacements), axis=1),
            cube_size=self._cube_size,
            buffer_size=self._buffer_size,
            voxel_size=self._voxel_size,
            device=self._device
        )
        dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            collate_fn=collate_synthetic_tree,
        )
        batch_points: List[np.ndarray] = []
        batch_displacements: List[np.ndarray] = []
        for batch in dataloader:
            input, displacements, mask = batch
            outputs = self._model(input)
            pred_displacements: torch.Tensor = outputs["direction"] * outputs["distance"]
            pred_displacements =  pred_displacements[mask]
            points: torch.Tensor = input.features[mask]
            displacements = displacements[mask]
            batch_points.append((points+pred_displacements).detach().cpu().numpy())
            batch_displacements.append((pred_displacements+displacements).detach().cpu().numpy())
        contracted_points: np.ndarray = np.concatenate(batch_points, axis=0)
        displacements: np.ndarray = np.concatenate(batch_displacements, axis=0)
        
        if self._chamfer_distance is not None:
            P: o3d.geometry.PointCloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self._contracted_points))
            Q: o3d.geometry.PointCloud= o3d.geometry.PointCloud(o3d.utility.Vector3dVector(contracted_points))

            min_bound = np.minimum(P.get_min_bound(), Q.get_min_bound())
            max_bound = np.maximum(P.get_max_bound(), Q.get_max_bound())

            # Reduce size imbalance and use the same voxel space
            P = P.voxel_down_sample_and_trace(self._voxel_size, min_bound, max_bound)[0]
            Q = Q.voxel_down_sample_and_trace(self._voxel_size, min_bound, max_bound)[0]
            chamfer_distance: float = np.mean(np.square(np.asarray(P.compute_point_cloud_distance(Q)))) + np.mean(np.square(np.asarray(Q.compute_point_cloud_distance(P))))
            if chamfer_distance > self._chamfer_distance:
                if not self._verbose:
                    return
                self._early_stopped = True
                return f"Early stopped at the {self._epoch}th contraction."
            self._chamfer_distance = chamfer_distance

        self._contracted_points = contracted_points
        self._displacements = displacements


        if not self._verbose:
            return
        
        contracted_pcd = o3d.geometry.PointCloud()
        contracted_pcd.points = o3d.utility.Vector3dVector(self._contracted_points)

        return f"Completed {self._epoch} contraction(s). {len(self._contracted_points)} point(s) remained.", contracted_pcd
    
    def get_pipeline(self):
        return [
            self._contract
        ] * self._max_iteration
    
    def output(self):
        return self._contracted_points, self._displacements

registry: Dict[str, CoreAlgorithmBase] = {
    "sparse_conv_based_contraction": SparseConvBasedContraction
}