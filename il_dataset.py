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

ACTION  (label — dict of arms, e.g. {"arm1": ...}; ONE arm for now, add "arm2" later with NO
         schema change — every reader iterates over whatever arms are present).
  Two student heads, two output frames (decoupled on purpose):
    GRASP head  → grab_u, grab_v   float   UV the teacher chose to grab  (frame-free, source-agnostic)
    DRAG  head  → release [x,y,z]   float   where the grabbed point is dragged to and released
                  path  [[x,y,z]*K] float   K intermediate waypoints between grab and release
  release/path FRAME: XY relative to the cloud centroid, Z absolute (height above the table, z=0
  at table). Centroid-relative XY is translation-invariant and matches the centroid-normalised
  state input; absolute Z matches the z_abs student feature. world = centroid + (x,y); z as-is.
  GRASP orientation:
    grab_quat [x,y,z,w] float  world quaternion of the grasp/wrist frame. Convention: identity
                  [0,0,0,1] = tool-Z pointing straight DOWN (−world Z), i.e. a vertical approach.
                  Mouse demos are always vertical-down → identity; VR demos fill in real wrist
                  rotations against this same convention, so the two sources merge unchanged.
  reference (not a head target):
    grab_xyz [x,y,z] float   world grab position    (for execution / debugging)
    grab_pcd_idx     int     resolved point index   (sim only; -1 when N/A, e.g. VR)
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
_ARM_KEYS   = ("grab_u", "grab_v", "release", "path", "grab_quat", "grab_xyz", "grab_pcd_idx", "reasoning")
IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]   # grasp pointing straight down (vertical approach); see ACTION docstring
PATH_LEN    = 5   # default sampling resolution for SYNTHETIC path generators (mouse teleop, scripted
                  # teacher) in head_RL.py — NOT a writer pad; the writer keeps paths variable-length
MAX_WP      = 8   # cap on transformer drag waypoints (release + up to MAX_WP-1 path pts); a knob, not a fix


def _new_id(source):
    global _counter
    with _lock:
        _counter += 1
        return f"{source}_{int(time.time()*1000)}_{_counter:04d}"


# ── writing ────────────────────────────────────────────────────────────────────────────────
def _clean_path(path):
    """Sanitise a recorded drag path: float-coerce each [x,y,z] and keep it VARIABLE-LENGTH (no
    padding) — the recording cadence (VR teleop samples one waypoint per second) is preserved as-is,
    so a long move loses no information and a short one isn't padded with fakes. Capped to MAX_WP-1
    intermediate points so path + release ≤ MAX_WP transformer waypoints (the release is never
    truncated away). Empty path = a direct grab→release straight-line drag (resolved at execution)."""
    pts = [[float(c) for c in p] for p in (path or [])]
    return pts[:MAX_WP - 1]


def _clean_quats(quats, n):
    """Sanitise a list of [x,y,z,w] wrist quaternions to exactly n entries (the # of waypoints they
    annotate): float-coerce, pad/truncate with identity. None → all identity (no orientation given)."""
    out = [list(IDENTITY_QUAT) if q is None else [float(c) for c in q] for q in (quats or [])]
    out = out[:n] + [list(IDENTITY_QUAT)] * max(0, n - len(out))
    return out


def make_arm(grab_u, grab_v, release, path, grab_quat=None, grab_xyz=None, grab_pcd_idx=-1,
             reasoning="", path_quat=None, release_quat=None):
    """Build one arm's action label. `release` is [x,y,z] (XY rel centroid, Z abs); `path` is a
    VARIABLE-LENGTH list of intermediate [x,y,z] waypoints (same frame) — VR/teleop callers sample it
    at 1 Hz; it is stored as given (capped to MAX_WP-1, not padded).
    `grab_quat` is the [x,y,z,w] grasp orientation (default identity = vertical-down; see ACTION
    docstring). `path_quat`/`release_quat` are the [x,y,z,w] WRIST orientations at each path waypoint
    and at release — the per-waypoint rotation the student predicts (default identity; VR teleop fills
    them). `path_quat` is aligned 1:1 with the cleaned `path`. VR/RealSense callers pass grab_pcd_idx=-1."""
    quat = list(IDENTITY_QUAT) if grab_quat is None else [float(c) for c in grab_quat]
    cpath = _clean_path(path)
    rquat = list(IDENTITY_QUAT) if release_quat is None else [float(c) for c in release_quat]
    return {
        "grab_u": float(grab_u), "grab_v": float(grab_v),
        "release": [float(c) for c in release],
        "path": cpath,
        "grab_quat": quat,
        "path_quat": _clean_quats(path_quat, len(cpath)),
        "release_quat": rquat,
        "grab_xyz": list(map(float, grab_xyz)) if grab_xyz is not None else None,
        "grab_pcd_idx": int(grab_pcd_idx),
        "reasoning": str(reasoning or ""),
    }


def record_sample(out_dir, source, state, action, *,
                  episode=None, turn=None, reward=None,
                  reward_before=None, reward_after=None,
                  teacher_model=None, extra=None):
    """Append one IL sample. `state` is a dict of the 4 STATE arrays; `action` is
    {"arm1": make_arm(...)} (add "arm2" later, same shape). Returns the sample_id."""
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


def action_to_targets(action, state, max_wp=MAX_WP):
    """Convert a stored (action, state) into transformer-student training targets, per present arm:
      GRASP head → grab_idx   : index of the grabbed POINT in state["pcd_xyz"] (Categorical target).
      DRAG  head → waypoints  : (max_wp, 3) ordered trajectory, last active = release, rest = path
                   wp_quat     : (max_wp, 4) per-waypoint wrist orientation [x,y,z,w], aligned with
                                 waypoints (identity for padded/missing); geodesic-loss target
                   active      : (max_wp,)   1 for the k real waypoints, 0 after (length / stop target)
                   length      : k
    Trajectory = path... + [release] (XY rel centroid, Z abs), truncated to max_wp. `state` is needed
    to resolve the grasp to a point index: use grab_pcd_idx when present (sim), else nearest point to
    grab_xyz (world→centroid-rel), else nearest predicted-UV point (last resort). Iterates over
    whatever arms are present (1 now, 2 later)."""
    pts = np.asarray(state["pcd_xyz"], np.float32)
    cen = np.asarray(state["centroid"], np.float32)
    out = {}
    for arm in sorted(action):
        a  = action[arm]
        gi = a.get("grab_pcd_idx", -1)
        gi = int(gi) if gi is not None else -1
        if not (0 <= gi < len(pts)):
            gx = a.get("grab_xyz")
            if gx is not None:
                rel = np.asarray(gx, np.float32) - cen           # world → centroid-relative
                gi  = int(np.argmin(np.linalg.norm(pts - rel, axis=1)))
            else:
                uv  = np.asarray(state["uv_pred"], np.float32)
                gi  = int(np.argmin(np.linalg.norm(uv - [a["grab_u"], a["grab_v"]], axis=1)))

        path  = [[float(c) for c in p] for p in (a.get("path") or [])]
        pquat = _clean_quats(a.get("path_quat"), len(path))      # per path-waypoint orientation
        rquat = a.get("release_quat") or list(IDENTITY_QUAT)
        traj  = path + [[float(c) for c in a["release"]]]        # last waypoint = release
        quats = pquat + [[float(c) for c in rquat]]
        traj, quats = traj[:max_wp], quats[:max_wp]
        k = len(traj)
        wp  = np.zeros((max_wp, 3), np.float32); wp[:k]  = np.asarray(traj, np.float32)
        wq  = np.zeros((max_wp, 4), np.float32); wq[:, 3] = 1.0   # pad with identity quaternion
        wq[:k] = np.asarray(quats, np.float32)
        act = np.zeros((max_wp,), np.float32);   act[:k] = 1.0
        out[arm] = {"grab_idx": gi, "waypoints": wp, "wp_quat": wq, "active": act, "length": k}
    return out
