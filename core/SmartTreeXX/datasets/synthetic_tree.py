
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Optional, Dict, List, Union, Tuple, Set, Optional
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch import SparseConvTensor

def load_tree_data(
        tree_path: str, 
        *,
        use_point: bool = True,
        use_xyz_coordinate: bool = True, 
        use_displacement: bool = True, 
        use_color: bool = True, 
        use_label: bool = True
) -> Dict[str, np.ndarray]:
    points: np.ndarray
    colors: Optional[np.ndarray]
    displacements: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    with np.load(tree_path) as f:
        points = f["xyz"] if use_point else None  # X(-Z)Y
        colors = f["rgb"] if use_color else None
        displacements = f["vector" if "vector" in f.files else "medial_vector"] if use_displacement else None
        labels = f["class_l"] if use_label else None

    if use_point and use_xyz_coordinate:
        points[:, 2] = -points[:, 2]
        points[:, [0, 2, 1]] = points[:, [0, 1, 2]]
    if use_displacement and use_xyz_coordinate:
        displacements[:, 2] = -displacements[:, 2]
        displacements[:, [0, 2, 1]] = displacements[:, [0, 1, 2]]

    return dict([
        [k, v] for k, v in [
            ("point", points), ("color", colors), ("displacement", displacements), ("label", labels)
        ] if v is not None
    ])

class SyntheticTree:
    _train_files: List[str]
    _val_files: List[str]
    _test_files: List[str]

    _use_foliage: bool

    def __init__(
            self, 
            *,
            dataset_dir: str,
            use_foliage: bool = False
    ) -> None:
        with open(os.path.join(dataset_dir, 'split.json'), 'r') as f:
            data: Dict[str, List[str]] = json.load(f)
            path_prefix: str = os.path.join(dataset_dir, 'branches_foliage' if use_foliage else 'branches')

            self.train_files = [os.path.join(path_prefix, path) for path in data['train']]
            self.val_files = [os.path.join(path_prefix, path) for path in data['validation']]
            self.test_files = [os.path.join(path_prefix, path) for path in data['test']]
        
        self._use_foliage = use_foliage
        return
    
    def get_split(self, split: str = "train") -> Union[List[str]]:
        if split == "train":
            return self.train_files
        elif split == "val":
            return self.val_files
        elif split == "test":
            return self.test_files
        else:
            raise NotImplementedError
    
class SyntheticTreeDataset(Dataset):
    _paths: List[str]
    _cache: Optional[Dict[int, Dict[str, torch.Tensor]]]
    _use_xyz_coordinates: bool
    _use_patchwise_normalization: bool
    _cube_size: float
    _voxel_size: float
    _device: str

    def __init__(
            self, 
            paths: List[str], 
            *,
            use_cache: bool = False,
            use_xyz_coordinates: bool = True,
            cube_size: float = 4.0,
            voxel_size: float = 0.01,
            device: str = "cuda"
    ) -> None:
        super().__init__()
        self._paths = paths
        self._cache: Optional[Dict[str, torch.Tensor]] = {} if use_cache else None
        self._use_xyz_coordinates = use_xyz_coordinates
        self._cube_size = cube_size
        self._voxel_size = voxel_size
        self._device = device
        return

    def __len__(self) -> int:
        return len(self._paths)
    
    def __getitem__(self, id: int) -> Dict[str, torch.Tensor]:
        points: torch.Tensor
        displacements: torch.Tensor

        if self._cache is not None and id in self._cache:
            points, displacements = self._cache[id]
        else:
            data: Dict[str, torch.Tensor] = load_tree_data(
                self._paths[id], 
                use_point=True,
                use_xyz_coordinate=self._use_xyz_coordinates, 
                use_displacement=True,
                use_color=False,
                use_label=False
            )
            points = torch.from_numpy(data["point"]).type(torch.float32)
            displacements = torch.from_numpy(data["displacement"]).type(torch.float32)
            if self._cache is not None:
                self._cache[id] = (points, displacements)

        points = points.to(self._device)
        displacements = displacements.to(self._device)
        
        random_point_id: int = torch.randint(0, points.shape[0], (1,)).item()
        cube_center: torch.Tensor = points[random_point_id]

        min_diagonal: torch.Tensor = cube_center - self._cube_size / 2
        max_diagonal: torch.Tensor = cube_center + self._cube_size / 2

        mask: torch.Tensor = torch.logical_and(
            points >= min_diagonal, points <= max_diagonal
        ).all(dim=1)

        points = points[mask]
        displacements = displacements[mask]

        min_diagonal = min_diagonal.squeeze()
        max_diagonal = max_diagonal.squeeze()

        point_ids: torch.Tensor = torch.arange(points.shape[0], device=points.device).unsqueeze(1)

        features: torch.Tensor = torch.cat([points, point_ids], dim=1)
        voxel_indices: torch.Tensor
        features, voxel_indices, _, _ = PointToVoxel(
            vsize_xyz=[self._voxel_size] * 3,
            coors_range_xyz=[
                min_diagonal[0], min_diagonal[1], min_diagonal[2],
                max_diagonal[0], max_diagonal[1], max_diagonal[2]
            ],
            num_point_features=features.shape[1],
            max_num_voxels=features.shape[0],
            max_num_points_per_voxel=1,
            device=features.device
        ).generate_voxel_with_id(features)

        features = features.squeeze(1)
        points = features[: ,:3]
        point_ids = features[:, 3].int()
        displacements = displacements[point_ids]

        batch_ids: torch.Tensor =  torch.zeros(
            (voxel_indices.shape[0], 1),
            dtype=voxel_indices.dtype,
            device=voxel_indices.device
        )
        voxel_indices = torch.cat([batch_ids, voxel_indices], dim=1)
        return points, displacements, voxel_indices

def collate_synthetic_tree(
        batch: Union[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]
) -> Tuple[SparseConvTensor, torch.Tensor, torch.Tensor]:
    batch_points: Union[List[torch.Tensor], torch.Tensor]
    batch_displacements: Union[List[torch.Tensor], torch.Tensor]
    batch_voxel_indices: Union[List[torch.Tensor], torch.Tensor]
    batch_mask: Union[Union[List[torch.Tensor], torch.Tensor], None] = None
    
    if len(batch[0]) == 3:
        batch_points, batch_displacements, batch_voxel_indices = zip(*batch)
    elif len(batch[0]) == 4:
        batch_points, batch_displacements, batch_voxel_indices, batch_mask = zip(*batch)
    
    for i, voxel_indices in enumerate(batch_voxel_indices):
        voxel_indices[:, 0] = i

    batch_points = torch.cat(batch_points)
    batch_displacements = torch.cat(batch_displacements)
    batch_voxel_indices = torch.cat(batch_voxel_indices)

    batch_size: int = len(batch_voxel_indices)

    input: SparseConvTensor = SparseConvTensor(
        features=batch_points,
        indices=batch_voxel_indices,
        spatial_shape=torch.max(batch_voxel_indices, dim=0)[0][1:],
        batch_size=batch_size
    )

    if batch_mask is not None:
        batch_mask = torch.cat(batch_mask)
        return input, batch_displacements, batch_mask
    return input, batch_displacements

class SingleTreeDataset(Dataset):
    _data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

    def __init__(
            self,
            tree_path_or_points_with_displacement: Union[str, np.ndarray],
            *,
            use_xyz_coordinates: bool = True,
            cube_size: float = 4.0,
            buffer_size: float = 0.4,
            voxel_size: float = 0.01,
            min_num_points: int = 1,
            device: str = "cuda"
    ) -> None:
        super().__init__()
        points: torch.Tensor
        displacements: torch.Tensor

        if isinstance(tree_path_or_points_with_displacement, str):
            if tree_path_or_points_with_displacement.endswith(".npz"):
                data: Dict[str, torch.Tensor] = load_tree_data(
                    tree_path=tree_path_or_points_with_displacement,
                    use_point=True,
                    use_xyz_coordinate=use_xyz_coordinates, 
                    use_displacement=True,
                    use_color=False,
                    use_label=False
                )
                points = torch.from_numpy(data["point"]).type(torch.float32)
                displacements = torch.from_numpy(data["displacement"]).type(torch.float32)
            else:
                raise ValueError
        else:
            points = torch.from_numpy(tree_path_or_points_with_displacement[:, :3])
            displacements = torch.from_numpy(tree_path_or_points_with_displacement[:, 3:])

        cube_indices: torch.Tensor = torch.div(points, cube_size, rounding_mode = "floor")
        point_counts_per_cube: torch.Tensor
        cube_indices, point_counts_per_cube = torch.unique(cube_indices, return_counts=True, dim=0)
        cube_centers: torch.Tensor = cube_indices * cube_size + cube_size / 2

        self._data = []
        for i in range(len(cube_centers)):
            if point_counts_per_cube[i] < min_num_points:
                continue
            sample_points: torch.Tensor
            sample_displacements: torch.Tensor
            sample_mask: torch.Tensor

            cube_center: torch.Tensor = cube_centers[i]
            min_diagonal: torch.Tensor = cube_center - cube_size / 2
            max_diagonal: torch.Tensor = cube_center + cube_size / 2
            sample_min_diagonal = min_diagonal - buffer_size
            sample_max_diagonal = max_diagonal + buffer_size
            mask: torch.Tensor = torch.logical_and(
                points >= sample_min_diagonal, points <= sample_max_diagonal
            ).all(dim=1)

            sample_points = points[mask]
            sample_displacements = displacements[mask]
            sample_mask = torch.logical_and(
                sample_points >= min_diagonal, sample_points <= max_diagonal
            ).all(dim=1)
            

            self._data.append((
                sample_points, 
                sample_displacements, 
                sample_min_diagonal.squeeze(), 
                sample_max_diagonal.squeeze(), 
                sample_mask
            ))

        self._voxel_size = voxel_size
        self._device = device
        return
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        points: torch.Tensor
        displacements: torch.Tensor
        min_diagonal: torch.Tensor
        max_diagonal: torch.Tensor
        mask: torch.Tensor

        points, displacements, min_diagonal, max_diagonal, mask = self._data[index]

        points = points.to(self._device)
        displacements = displacements.to(self._device)
        mask = mask.to(self._device)

        point_ids: torch.Tensor = torch.arange(points.shape[0], device=points.device).unsqueeze(1)
        features: torch.Tensor = torch.cat([points, point_ids], dim=1)
         
        voxel_indices: torch.Tensor
        features, voxel_indices, _, _ = PointToVoxel(
            vsize_xyz=[self._voxel_size] * 3,
            coors_range_xyz=[
                min_diagonal[0], min_diagonal[1], min_diagonal[2],
                max_diagonal[0], max_diagonal[1], max_diagonal[2]
            ],
            num_point_features=features.shape[1],
            max_num_voxels=features.shape[0],
            max_num_points_per_voxel=1,
            device=features.device
        ).generate_voxel_with_id(features)
        
        features = features.squeeze(1)
        points = features[:, :3]
        point_ids = features[:, 3].int()
        displacements = displacements[point_ids]
        mask = mask[point_ids]

        batch_ids: torch.Tensor =  torch.zeros(
            (voxel_indices.shape[0], 1),
            dtype=voxel_indices.dtype,
            device=voxel_indices.device
        )
        voxel_indices = torch.cat([batch_ids, voxel_indices], dim=1)

        return points, displacements, voxel_indices, mask
