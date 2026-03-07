import os
import subprocess
import traceback
import sys

def _update_requirements(root_dir: str):
    python_exe = sys.executable
    pip_install_success = True
    req_file = os.path.join(root_dir, "requirements.txt")
    packages = ["-r", req_file]

    cmd = [
        python_exe, "-m", "pip", "install", "-U", 
        "--no-cache-dir"
    ] + packages
    
    subprocess.run(
        cmd,
        stdout=None,       
        stderr=None,    
        text=True,
        encoding="utf-8",
        check=True,       
        shell=False     
    )

    print("\n--------------------------------------------------")
    print("Pip installation completed successfully!")

    return pip_install_success

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
            os.remove(os.path.join(root_dir, useless_file_path))
        except FileNotFoundError:
            pass

    
    useless_directories = [
    ]
    
    for useless_dir in useless_directories:
        try:
            os.rmdir(os.path.join(root_dir, useless_dir))
        except FileNotFoundError:
            pass
        
    print("Clean finished!")
    
    # Update requirements
    _update_requirements(root_dir)

if __name__ == "__main__":
    try:
        run_post_install(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        traceback.print_exc()
        print("-----------------------")
        print("Please retry later.")