"""
il_dataset.py — unified imitation-learning sample store for the garment VLA.

ONE schema, multiple teachers. Haiku (now), human VR demos (later), or a sim policy all write
the SAME (state, action) tuple, so the student VLA trains on their union. The student input is
pure geometry + UV → it is source-agnostic (works on sim mesh clouds AND RealSense clouds).

Layout on disk:
  <dir>/
    samples/<sample_id>.npz     state arrays (student input)
    index.jsonl                 one JSON line per sample: action label + metadata + npz path

STATE  (student input — geometry only, no mesh-vertex identity, so it transfers to RealSense):
  pcd_xyz   (N,3) float32   centroid-normalised point cloud
  normals   (N,3) float32   per-point normals
  uv_pred   (N,2) float32   per-point predicted UV from the UV Mapper
  centroid  (3,)  float32   world centroid that was subtracted

ACTION  (label — per arm "arm1","arm2"):
  grab_u, grab_v   float   UV the teacher chose to grab  (SOURCE-AGNOSTIC — primary target)
  grab_xyz [x,y,z] float   world grab position           (source-agnostic, for reference)
  grab_pcd_idx     int     resolved point index          (sim only; -1 when N/A, e.g. VR)
  dx, dy, dz       float   drag vector, world metres
  reasoning        str     teacher's explanation (Haiku); "" for VR

METADATA: source, episode, turn, ts, reward, reward_before, reward_after, teacher_model

Why grab is UV (not a point index): a RealSense cloud has no correspondence to mesh vertices,
so a point index is meaningless across sources. UV is assigned per-point by the UV Mapper for
any cloud, so grab_u/grab_v is the one grab representation that works in sim AND on real data.
"""
import os, json, time, threading
import numpy as np

_lock = threading.Lock()
_counter = 0

_STATE_KEYS = ("pcd_xyz", "normals", "uv_pred", "centroid")
_ARM_KEYS   = ("grab_u", "grab_v", "grab_xyz", "grab_pcd_idx", "dx", "dy", "dz", "reasoning")


def _new_id(source):
    global _counter
    with _lock:
        _counter += 1
        return f"{source}_{int(time.time()*1000)}_{_counter:04d}"


# ── writing ────────────────────────────────────────────────────────────────────────────────
def make_arm(grab_u, grab_v, dx, dy, dz, grab_xyz=None, grab_pcd_idx=-1, reasoning=""):
    """Build one arm's action label. VR/RealSense callers pass grab_pcd_idx=-1."""
    return {
        "grab_u": float(grab_u), "grab_v": float(grab_v),
        "grab_xyz": list(map(float, grab_xyz)) if grab_xyz is not None else None,
        "grab_pcd_idx": int(grab_pcd_idx),
        "dx": float(dx), "dy": float(dy), "dz": float(dz),
        "reasoning": str(reasoning or ""),
    }


def record_sample(out_dir, source, state, action, *,
                  episode=None, turn=None, reward=None,
                  reward_before=None, reward_after=None,
                  teacher_model=None, extra=None):
    """Append one IL sample. `state` is a dict of the 4 STATE arrays; `action` is
    {"arm1": make_arm(...), "arm2": make_arm(...)}. Returns the sample_id."""
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sid = _new_id(source)
    npz_path = os.path.join(samples_dir, sid + ".npz")
    np.savez(npz_path, **{k: np.asarray(state[k], np.float32) for k in _STATE_KEYS})

    row = {
        "sample_id": sid,
        "npz": os.path.relpath(npz_path, out_dir),
        "source": source,
        "episode": episode, "turn": turn,
        "ts": time.time(),
        "reward": reward,
        "reward_before": reward_before,
        "reward_after": reward_after,
        "improved": (None if reward_before is None or reward_after is None
                     else bool(reward_after > reward_before)),
        "teacher_model": teacher_model,
        "n_points": int(len(state["pcd_xyz"])),
        "action": action,
    }
    if extra:
        row.update(extra)
    with _lock:
        with open(os.path.join(out_dir, "index.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return sid


def count(out_dir):
    idx = os.path.join(out_dir, "index.jsonl")
    if not os.path.exists(idx):
        return 0
    with open(idx) as fh:
        return sum(1 for ln in fh if ln.strip())


# ── reading (training side) ──────────────────────────────────────────────────────────────────
def load_index(out_dir):
    """Return list of all sample rows (action + metadata). Mergeable across sources."""
    rows = []
    with open(os.path.join(out_dir, "index.jsonl")) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def load_sample(out_dir, row):
    """Load one sample → (state dict of arrays, action dict)."""
    d = np.load(os.path.join(out_dir, row["npz"]))
    state = {k: d[k] for k in _STATE_KEYS}
    return state, row["action"]


# ── student I/O spec: exactly what the VLA reads in, and what it predicts ──────────────────────
def featurize(state):
    """Per-point input tensor the student VLA consumes: (N, 9) =
    [x, y, z, u, v, nx, ny, nz, z_abs]. Source-agnostic (sim mesh cloud or RealSense cloud)."""
    xyz   = np.asarray(state["pcd_xyz"], np.float32)
    uv    = np.asarray(state["uv_pred"], np.float32)
    nrm   = np.asarray(state["normals"], np.float32)
    z_abs = (xyz[:, 2] + np.asarray(state["centroid"], np.float32)[2])[:, None]
    return np.concatenate([xyz, uv, nrm, z_abs], axis=1).astype(np.float32)


def action_to_targets(action, grid=32):
    """Convert a stored action into student training targets:
      per arm → grab UV-grid bin (0..grid*grid-1)  +  drag vector (dx,dy,dz).
    UV-grid classification keeps grabs multimodal AND source-agnostic (no point indices)."""
    out = {}
    for arm in ("arm1", "arm2"):
        a  = action[arm]
        bu = min(max(int(a["grab_u"] * grid), 0), grid - 1)
        bv = min(max(int(a["grab_v"] * grid), 0), grid - 1)
        out[arm] = {"grab_bin": bv * grid + bu, "drag": [a["dx"], a["dy"], a["dz"]]}
    return out
