"""Find images that show the same picture, or the same fixed camera, by perceptual hash.

dhash() is a 64-bit difference hash: shrink to 9x8 grey, compare neighbouring pixels. Two
images whose hashes differ in at most a few bits look alike. That catches two things:
  - the same picture saved twice (re-encoded, resized): same_picture() confirms these with
    a pixel check, so a scene from a fixed camera at another moment is not counted;
  - frames from the same fixed camera: camera_groups() links all look-alike frames, so a
    camera's frames stay on one side of a train/test split.

For fixed cameras over crowded rooms the 64-bit hash is too fine: students move between frames
and the hash of one camera's frames drifts by more than a few bits (on ClaBehavior, 4 bits split
13 camera views into 107 groups, 93 of them single frames). view_groups() therefore compares
whole thumbnails instead: two frames of the same size whose 48x27 grey thumbnails correlate at
min_corr or more share a camera, linked transitively. scan_view_groups() prints the number of
groups for a range of min_corr, so the threshold can be read off where the count stops changing.

Hashes are cached in a JSON file next to the data, so the images are read once.
"""
import json
import os

import numpy as np
from PIL import Image


def dhash(path):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hashes(paths, cache_path=None):
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        cache = json.load(open(cache_path))
    out = []
    for p in paths:
        if p not in cache:
            cache[p] = dhash(p)
        out.append(cache[p])
    if cache_path:
        json.dump(cache, open(cache_path, "w"))
    return np.array(out, dtype=np.uint64)


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x):
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    return _POPCOUNT8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1, dtype=np.uint8)


def near_pairs(h, max_bits=4, block=256):
    """Index pairs (i < j) whose hashes differ in at most max_bits bits. Works in blocks of rows,
    so 10,000 images need only a few tens of MB."""
    h = np.asarray(h, dtype=np.uint64)
    pairs = []
    for a in range(0, len(h), block):
        bits = _popcount64(np.bitwise_xor(h[a:a + block, None], h[None, :]))
        ii, jj = np.nonzero(bits <= max_bits)
        ii = ii + a
        keep = ii < jj
        pairs += list(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


def _thumb(path, size=64):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)


def same_picture(path_a, path_b, max_mean_diff=6.0, _memo=None):
    """True if two look-alike images are the same picture (not the same camera at another moment):
    mean absolute difference of the 64x64 grey thumbnails below max_mean_diff (0-255 scale)."""
    memo = _memo if _memo is not None else {}
    for p in (path_a, path_b):
        if p not in memo:
            memo[p] = _thumb(p)
    return float(np.abs(memo[path_a] - memo[path_b]).mean()) < max_mean_diff


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def labels(self):
        roots = {}
        return [roots.setdefault(self.find(i), len(roots)) for i in range(len(self.p))]


def camera_groups(paths, max_bits=4, cache_path=None):
    """Group frames from the same fixed camera: link frames whose hashes differ in at most max_bits
    bits, then take connected components. Returns one group number per path."""
    h = hashes(paths, cache_path)
    uf = UnionFind(len(paths))
    for i, j in near_pairs(h, max_bits):
        uf.union(i, j)
    return uf.labels()


def same_picture_pairs(paths, max_bits=4, cache_path=None):
    """Pairs of paths that are the same picture saved twice (hash look-alike + pixel check)."""
    h = hashes(paths, cache_path)
    memo = {}
    return [(paths[i], paths[j]) for i, j in near_pairs(h, max_bits) if same_picture(paths[i], paths[j], _memo=memo)]


def view_signature(path, size=(48, 27)):
    """Grey thumbnail, mean 0 and unit norm, plus the image size: frames of one fixed camera share the
    background (walls, board, desks), which dominates a thumbnail even with students moving."""
    with Image.open(path) as im:
        wh = im.size
        g = np.asarray(im.convert("L").resize(size, Image.BILINEAR), dtype=np.float32).ravel()
    g -= g.mean()
    n = float(np.linalg.norm(g))
    return wh, (g / n if n > 0 else g)


def view_similarity(paths, cache=None):
    """Correlation matrix of the view signatures (-1 if the image sizes differ), and the sizes."""
    memo = cache if cache is not None else {}
    sigs = []
    for p in paths:
        if p not in memo:
            memo[p] = view_signature(p)
        sigs.append(memo[p])
    sizes = [wh for wh, _ in sigs]
    m = np.stack([v for _, v in sigs])
    corr = m @ m.T
    same = np.array([[a == b for b in sizes] for a in sizes])
    corr[~same] = -1.0
    return corr, sizes


def view_groups(paths, min_corr=0.8, corr=None):
    """One camera-view group number per path: same-size frames whose thumbnails correlate at
    min_corr or more are linked, then connected components are taken."""
    if corr is None:
        corr, _ = view_similarity(paths)
    uf = UnionFind(len(paths))
    ii, jj = np.nonzero(np.triu(corr >= min_corr, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.union(i, j)
    return uf.labels()


def scan_view_groups(paths, thresholds=(0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4)):
    """[(min_corr, number of groups, single-frame groups, sizes)] for each threshold."""
    corr, _ = view_similarity(paths)
    out = []
    for t in thresholds:
        g = view_groups(paths, t, corr)
        sizes = sorted(np.bincount(g).tolist(), reverse=True)
        out.append((t, len(sizes), sum(1 for x in sizes if x == 1), sizes))
    return out
