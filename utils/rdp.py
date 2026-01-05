import numpy as np


def _max_dist_segment(pts, start, end):
    if end - start <= 1:
        return 0.0, -1

    a = pts[start]
    b = pts[end]
    seg = b - a
    seg_len2 = np.dot(seg, seg)
    if seg_len2 == 0.0:
        mid = pts[start + 1:end]
        if mid.size == 0:
            return 0.0, -1
        diff = mid - a
        dists = np.linalg.norm(diff, axis=1)
        max_idx_in = np.argmax(dists)
        return dists[max_idx_in], start + 1 + max_idx_in

    mid = pts[start + 1:end]  
    ap = mid - a                 
    t = (ap @ seg) / seg_len2 
    t = np.clip(t, 0.0, 1.0)

    proj = a + t[:, None] * seg 
    diff = mid - proj
    dists = np.linalg.norm(diff, axis=1)

    max_idx_in = np.argmax(dists)
    max_dist = dists[max_idx_in]
    max_index = start + 1 + max_idx_in
    return float(max_dist), int(max_index)


def rdp_fast(points, epsilon):
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n <= 2:
        return pts.copy(), list(range(n))

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, n - 1)]

    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue

        max_dist, max_index = _max_dist_segment(pts, start, end)

        if max_dist > epsilon:
            keep[max_index] = True
            stack.append((start, max_index))
            stack.append((max_index, end))

    keep_indices = np.nonzero(keep)[0].tolist()
    simplified = pts[keep]
    return simplified, keep_indices
