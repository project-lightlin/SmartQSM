import os
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
os.remove(os.path.join(current_dir, "configs/layerwise-clustering-FEWER-LEAVES.yaml"))
os.remove(os.path.join(current_dir, "configs/layerwise-clustering-MORE-LEAVES.yaml"))
