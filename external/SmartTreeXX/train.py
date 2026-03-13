
import sys
import os
ROOT_DIR =os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import yaml
from datasets.synthetic_tree import SyntheticTree, SingleTreeDataset, SyntheticTreeDataset, collate_synthetic_tree
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import torch
from models.smart_tree_xx import SmartTreeXX
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
from metrics.displacement_tracker import DisplacementTracker
from spconv.pytorch import SparseConvTensor
import torch.nn.functional as F
import shutil

def main():
    print("Initializing...")
    with open(os.path.join(ROOT_DIR, "configs/train_smart_tree_xx.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Parameters
    project: str = config["project"]
    dataset_dir: str = config["dataset_dir"]
    use_foliage: bool = config["use_foliage"]
    use_log_distance: bool = config["use_log_distance"]
    device: str = config['device']
    seed: int = config['seed']
    lr: float = config['lr']
    max_epoch: int = config['max_epoch']
    max_patience: int = config['max_patience']
    cube_size_for_train: float = config['train']['cube_size']
    voxel_size_for_train: float = config['train']['voxel_size']
    batch_size_for_train: float = config['train']['batch_size']
    shuffle_for_train: bool = config['train']['shuffle']
    use_cache_for_val: bool = config['val']['use_cache']
    cube_size_for_val: float = config['val']['cube_size']
    buffer_size_for_val: int = config['val']['buffer_size']
    voxel_size_for_val: float = config['val']['voxel_size']
    batch_size_for_val: int = config['val']['batch_size']
    u_channels_for_model: int = config['model']['u_channels']
    mlp_hidden_layers_for_model: List[int] = config['model']['mlp_hidden_layers']
    conv_block_reps_for_model: int = config['model']['conv_block_reps']

    # Training set
    synthetic_tree: SyntheticTree = SyntheticTree(
        dataset_dir=dataset_dir,
        use_foliage=use_foliage
    )

    train_files: List[str] = synthetic_tree.get_split("train")
    train_dataset: SyntheticTreeDataset = SyntheticTreeDataset(
        train_files,
        use_cache=False,
        use_xyz_coordinates=True,
        cube_size=cube_size_for_train,
        voxel_size=voxel_size_for_train,
        device=device
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_for_train,
        shuffle=shuffle_for_train,
        collate_fn=collate_synthetic_tree
    )

    val_files: List[str] = synthetic_tree.get_split("val")
    val_cache: Dict[str, DataLoader] = {}

    # Model
    cfg: Dict[str, Any] = {
        "u_channels": u_channels_for_model,
        "mlp_hidden_layers": mlp_hidden_layers_for_model,
        "conv_block_reps": conv_block_reps_for_model,
    }
    model: SmartTreeXX = SmartTreeXX(
        **cfg
    ).to(device)

    # Optimizer, scheduler
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    # Log
    log_dir: str = os.path.join(ROOT_DIR, f"logs/{project}/{datetime.now().strftime("%y%m%d_%H%M%S")}")
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(
        os.path.join(ROOT_DIR, "configs/train_smart_tree_xx.yaml"),
        os.path.join(log_dir, "train_config_copy.yaml")
    )
    ckpt_dir: str = os.path.join(log_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    summary_writer: SummaryWriter = SummaryWriter(log_dir)
    
    # Other
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    patience: int = 0
    min_val_loss: float = float("inf")
    epoch_with_min_val_loss: int = 0
    prev_val_loss: float = float("inf")
    displacement_tracker: DisplacementTracker = DisplacementTracker(use_log_distance=use_log_distance)

    # Variables
    input: SparseConvTensor
    pred_directions: torch.Tensor
    pred_distances: torch.Tensor
    gt_displacements: torch.Tensor
    gt_directions: torch.Tensor
    gt_distances: torch.Tensor
    mask: torch.Tensor
    early_stopped: bool = False

    print(f"Project {project} will be logged to {log_dir}.")
    start_time: datetime = datetime.now()
    pbar: tqdm = tqdm(range(1, max_epoch + 1), desc="Epoch 1")
    for epoch in pbar:
        # Train
        displacement_tracker.clear()
        model.train()
        for batch in tqdm(train_dataloader, desc="Training", leave=False, total=len(train_dataloader)):
            input, gt_displacements = batch
            outputs: Dict[str, torch.Tensor] = model.forward(input)
            pred_directions = outputs["direction"]
            pred_distances = outputs["distance"]
            gt_directions = F.normalize(gt_displacements)
            gt_distances = torch.norm(gt_displacements, dim=1, keepdim=True)
            loss: torch.Tensor = displacement_tracker.track(pred_directions=pred_directions, gt_directions=gt_directions, pred_distances=pred_distances, gt_distances=gt_distances)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for key, value in displacement_tracker.get_metrics().items():
            summary_writer.add_scalar("TRAIN-"+key, value, epoch)
        
        # Evaluate
        displacement_tracker.clear()
        model.eval()
        with torch.no_grad():
            for val_file in tqdm(val_files, desc="Evaluating", leave=False):
                eval_dataloader: DataLoader
                if val_file not in val_cache:
                    eval_dataloader = DataLoader(
                        SingleTreeDataset(
                            val_file, 
                            use_xyz_coordinates=True, 
                            cube_size=cube_size_for_val, 
                            buffer_size=buffer_size_for_val,
                            voxel_size=voxel_size_for_val, 
                            device=device,
                            min_num_points=20
                        ),
                        batch_size=batch_size_for_val,
                        shuffle=False,
                        collate_fn=collate_synthetic_tree
                    )
                    if use_cache_for_val:
                        val_cache[val_file] = eval_dataloader
                else:
                    eval_dataloader = val_cache[val_file]
                for batch in eval_dataloader:
                    input, gt_displacements, mask = batch
                    outputs: Dict[str, torch.Tensor] = model.forward(input)
                    pred_directions = outputs["direction"]
                    pred_distances = outputs["distance"]
                    gt_directions = F.normalize(gt_displacements)
                    gt_distances = torch.norm(gt_displacements, dim=1, keepdim=True)
                    displacement_tracker.track(pred_directions=pred_directions, gt_directions=gt_directions, pred_distances=pred_distances, gt_distances=gt_distances, mask=mask)
        val_loss: float
        for key, value in displacement_tracker.get_metrics().items():
            summary_writer.add_scalar("VAL-"+key, value, epoch)
            if key == "loss":
                val_loss = value
        scheduler.step(val_loss)
        prev_val_loss = val_loss
        
        # Early stopping & model save
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epoch_with_min_val_loss = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "hparams": cfg

                },
                os.path.join(ckpt_dir, f"epoch_{epoch}={val_loss:.4f}.pth")
            )
        else:
            patience += 1
            if patience >= max_patience:
                early_stopped = True
                pbar.close()
                break
        
        # Update display
        pbar.set_postfix({
            "Prev.Loss": f"{prev_val_loss:.4f}", 
            "MinLoss":f"{min_val_loss:.4f}", 
            "atE.": epoch_with_min_val_loss
        })
        pbar.set_description(f"Epoch {epoch + 1}")
    
    # Result
    if early_stopped:
        print(f"Early stopped at epoch {epoch}.")
    end_time: datetime = datetime.now()
    print(f"Finished. Elapsed time: {(end_time - start_time).total_seconds()} sec(s).")
    return

if __name__ == "__main__":
    main()