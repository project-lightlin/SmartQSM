import os
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

# 1.4.5
try:
    os.remove(os.path.join(current_dir, "configs/layerwise-clustering-FEWER-LEAVES.yaml"))
except FileNotFoundError:
    pass
try:
    os.remove(os.path.join(current_dir, "configs/layerwise-clustering-MORE-LEAVES.yaml"))
except FileNotFoundError:
    pass
print("Cleaned outdated configs.")