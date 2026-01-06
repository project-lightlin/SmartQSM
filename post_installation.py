import os

def run_post_install(root_dir: str):
    # Remove official useless files
    useless_file_paths = [
        "configs/layerwise-clustering-FEWER-LEAVES.yaml",
        "configs/layerwise-clustering-MORE-LEAVES.yaml",
        "configs/layerwise-clustering-FEW-LEAVES.yaml",
        "configs/layerwise-clustering-SOME-LEAVES.yaml",
        "configs/layerwise-clustering-MANY-LEAVES.yaml",
        "configs/spconv-contraction-FEW-LEAVES-cpu.yaml",
        "configs/spconv-contraction-FEW-LEAVES-cuda.yaml",
        "configs/spconv-contraction-SOME-LEAVES-cpu.yaml",
        "configs/spconv-contraction-SOME-LEAVES-cuda.yaml",
        "configs/spconv-contraction-MANY-LEAVES-cpu.yaml",
        "configs/spconv-contraction-MANY-LEAVES-cuda.yaml"
    ]
    for useless_file_path in useless_file_paths:
        try:
            os.remove(os.path.join(current_dir, useless_file_path))
        except FileNotFoundError:
            pass
