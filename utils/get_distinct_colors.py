from distinctipy._colorsets_data import colors
from typing import List, Tuple

def get_distinct_colors(num_colors: int, colormap: str = "normal") -> List[Tuple[float, float, float]]:
    if colormap not in colors:
        raise ValueError(f"colormap {colormap} not found")
    color_count: int = len(colors[colormap])
    colors_: List[Tuple[float, float, float]] = []
    for i in range(num_colors):
        colors_.append(colors[colormap][i % color_count])
    return colors_