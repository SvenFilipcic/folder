"""
grasp_regions.py — ordered canonical grab-region config for the deterministic UV teacher.

A *region* is a named patch in canonical UV space (`uv_center`, `uv_radius`) plus the flat-layout
position it must end up at (`target_off`). You annotate the regions ONCE on the canonical garment
with `data/label_grasp_regions.py`; the teacher (`head_RL.py --scripted`) then replays them, in
priority order, on every fresh crumple:

    for region in regions (priority order):
        grab the visible point whose PREDICTED UV is nearest region.uv_center
        drag it to region.target_off ; once the region sits within tol of its target → next region

WHY this works without a learned policy: the UV Mapper already answers "where does this fabric
belong" (its predicted UV), and the flat reference answers "where is that on the table". The region
list only adds the human prior of *which parts to pull first* (e.g. pin the body, then the sleeves,
then the hem) — exactly the few-shot keypoint sequence UniGarmentManip uses.

FRAME: `target_off` is stored CENTROID-RELATIVE — it is the (dx,dy) of the region's flat position
measured from the flat layout's own centre, plus the table z. That is precisely the action frame
`_execute_drag_path` expects (XY relative to the cloud centroid, Z absolute, table=0), so a region
target drops in as a `release` with no extra transform. Fixed flat orientation ⇒ rotation-variant
target (the garment is turned INTO this pose), matching the white outline drawn for Haiku.
"""
import os
import json
import numpy as np

_ROOT        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_ROOT, "reference", "grasp_regions.json")
_REF_GRAPH   = os.path.join(_ROOT, "reference", "majca_mesh_graph.npz")
_REF_FLAT    = os.path.join(_ROOT, "reference", "majca_flat_reference_uv.npz")


def load_reference():
    """Canonical per-vertex arrays: node_uv (N,2), node_panel (N,), flat_pts (N,3) world XY+Z of the
    garment laid perfectly flat. Same vertex indexing across all three."""
    g  = np.load(_REF_GRAPH)
    fr = np.load(_REF_FLAT)
    return (g["node_uv"].astype(np.float32),
            g["node_panel"].astype(np.int32),
            fr["points"].astype(np.float32))


def target_offset(uv_center, uv_radius, node_uv=None, flat_pts=None):
    """Centroid-relative flat target [dx, dy, z] of the canonical verts within `uv_radius` of
    `uv_center` in UV space. (Front/back share UV and the same flat footprint, so panel is ignored —
    both layers map to the same table XY.)"""
    if node_uv is None or flat_pts is None:
        node_uv, _, flat_pts = load_reference()
    uc  = np.asarray(uv_center, np.float32)
    d   = np.linalg.norm(node_uv - uc, axis=1)
    sel = d <= float(uv_radius)
    if not sel.any():                                  # radius missed every vert → nearest one
        sel = d <= (d.min() + 1e-6)
    cen = flat_pts[:, :2].mean(0)                      # flat layout centre = action-frame origin
    off = flat_pts[sel, :2].mean(0) - cen
    z   = float(flat_pts[:, 2].mean())                 # table height
    return [float(off[0]), float(off[1]), z]


def make_region(name, uv_center, uv_radius, node_uv=None, flat_pts=None, panel=-1):
    """Build one region dict (with its computed flat target) for the config."""
    uc = [float(uv_center[0]), float(uv_center[1])]
    return {
        "name":       str(name),
        "uv_center":  uc,
        "uv_radius":  float(uv_radius),
        "panel":      int(panel),
        "target_off": target_offset(uc, uv_radius, node_uv, flat_pts),
    }


def save(regions, path=DEFAULT_PATH):
    with open(path, "w") as f:
        json.dump({"regions": list(regions)}, f, indent=2)
    return path


def load(path=DEFAULT_PATH):
    with open(path) as f:
        return json.load(f)["regions"]
