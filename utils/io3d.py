import open3d as o3d
from pathlib import Path
import laspy
import numpy as np
import ezdxf
from typing import Tuple
import pypcd4

def pcd_to_o3d(path: str) -> o3d.geometry.PointCloud:
    pc = pypcd4.PointCloud.from_path(path)
    data = pc.pc_data
    def _get_first_existing_field(names):
        for n in names:
            if n in pc.fields:
                return n
        return None

    x_name = _get_first_existing_field(["x"])
    y_name = _get_first_existing_field(["y"])
    z_name = _get_first_existing_field(["z"])
    if not all([x_name, y_name, z_name]):
        raise ValueError(f"Key fields x/y/z not found in PCD file: fields = {pc.fields}")

    x = data[x_name].astype(np.float64)
    y = data[y_name].astype(np.float64)
    z = data[z_name].astype(np.float64)
    xyz = np.stack([x, y, z], axis=-1) 

    colors = None

    if "rgb" in pc.fields:
        rgb_raw = data["rgb"]

        if rgb_raw.dtype == np.float32:
            rgb_uint32 = rgb_raw.view(np.uint32)
        elif rgb_raw.dtype == np.float64:
            rgb_uint32 = rgb_raw.astype(np.float32).view(np.uint32)
        else:
            rgb_uint32 = rgb_raw.view(np.uint32)

        r = (rgb_uint32 >> 16) & 0xFF
        g = (rgb_uint32 >> 8) & 0xFF
        b = rgb_uint32 & 0xFF

        rgb_255 = np.stack([r, g, b], axis=-1).astype(np.float32)
        colors = (rgb_255 / 255.0).astype(np.float64)

    elif {"r", "g", "b"}.issubset(set(pc.fields)):
        r = data["r"]
        g = data["g"]
        b = data["b"]

        def _normalize_channel(ch: np.ndarray) -> np.ndarray:
            ch = np.asarray(ch)
            assert int(ch.min()) >= 0, f"Color channel min value {ch.min()} is out of range."
            if np.issubdtype(ch.dtype, np.integer):
                if ch.dtype == np.uint8 or ch.dtype == np.int8:
                    return np.clip(ch / 255.0, 0.0, 1.0)
                elif ch.dtype == np.uint16 or ch.dtype == np.int16:
                    return np.clip(ch / 65535.0, 0.0, 1.0)
                else:
                    raise NotImplementedError(f"Unknown dtype for color channel: {ch.dtype}")
            else:
                maxv = float(ch.max()) if ch.size > 0 else 1.0
                if maxv <= 1.0:
                    return np.clip(ch.astype(np.float32), 0.0, 1.0)
                elif int(maxv) <= 255:
                    return np.clip(ch.astype(int) / 255.0, 0.0, 1.0)
                elif int(maxv) <= 65535:
                    return np.clip(ch.astype(int) / 65535.0, 0.0, 1.0)
                else:
                    raise ValueError(f"Color channel max value {maxv} is out of range.")

        r01 = _normalize_channel(r)
        g01 = _normalize_channel(g)
        b01 = _normalize_channel(b)

        colors = np.stack([r01, g01, b01], axis=-1).astype(np.float64)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None and colors.shape[0] == xyz.shape[0]:
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pcd_o3d

def read_point_cloud(cloud_path: str) -> Tuple[o3d.geometry.PointCloud, str]:
    suffix: str = Path(cloud_path).suffix
    cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    projection: str = ""
    
    if suffix in (".ply"):
        cloud = o3d.io.read_point_cloud(cloud_path)
    elif suffix in (".pcd"):
        cloud = pcd_to_o3d(cloud_path)
    elif suffix in (".las", ".laz"):
        las: laspy.LasData = laspy.read(cloud_path)
        try:
            projection = str(las.header.parse_crs())
        except Exception:
            pass
        cloud.points = o3d.utility.Vector3dVector(las.xyz)
        if hasattr(las, "red"):
            if las.red.dtype == np.uint16:
                cloud.colors = o3d.utility.Vector3dVector(np.stack((las.red, las.green, las.blue), axis=-1, dtype=np.float32) / 65535.0)
            else:
                cloud.colors = o3d.utility.Vector3dVector(np.stack((las.red, las.green, las.blue), axis=-1, dtype=np.float32) / 255.0)
    elif suffix in (".txt", ".csv", ".xyz", ".pts"): # .xyz to .pts might encounter decoding errors when using open3d.io.read_point_cloud
        data = np.loadtxt(
            cloud_path, 
            dtype=np.float64, 
            delimiter="," if suffix == ".csv" else None
        )
        cloud.points = o3d.utility.Vector3dVector(data[:, :3])
        if data.shape[1] > 3:
            
            if data.shape[1] == 6:
                rgb = data[:, 3:6]  # [N,3] float64
                min_val = rgb.min()
                max_val = rgb.max()
                if min_val > 0.0 and int(max_val) <= 65535:
                    if max_val <= 1.0:
                        colors_01 = np.clip(rgb, 0.0, 1.0)
                    elif int(max_val) <= 255:
                        colors_01 = np.clip(rgb.astype(int) / 255.0, 0.0, 1.0)
                    elif int(max_val) <= 65535:
                        colors_01 = np.clip(rgb.astype(int) / 65535.0, 0.0, 1.0)

                    cloud.colors = o3d.utility.Vector3dVector(colors_01.astype(np.float64))
    else:
        raise NotImplementedError("Unsupported file extension.")
    return cloud, projection

def write_polyline(lineset: o3d.geometry.LineSet, dxf_path: str) -> None:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for line in lineset.lines:
        p1, p2 = lineset.points[line[0]], lineset.points[line[1]]
        dxf_line = msp.add_line(p1, p2)
        dxf_line.dxf.color = 1
    doc.saveas(dxf_path)
    return