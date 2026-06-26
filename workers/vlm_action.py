"""
vlm_action.py — VLA: overhead UV-overlay image → ONE-arm grab+drag action via Claude Haiku.

Called by head_RL.py in VLA mode:
    python vlm_action.py --image rl_uv_overlay.png --out /tmp/vlm_action.json

head_RL.py renders an overhead view of the garment where every point is coloured by its
predicted UV (R=u, G=v). Haiku reads that image and picks (single arm for now):
  grab    (u,v)        — WHICH bit of fabric to grab, in UV space (frame-free, source-agnostic)
  release (x,y,z)      — WHERE to drag it to and let go
  path    [(x,y,z)*5]  — 5 waypoints the hand passes through between grab and release
xyz FRAME: origin = garment centre on the table, +x right, +y away from camera, +z up, metres.
(So x,y are relative to the cloud centroid; z is absolute height above the table.)
head_RL.py snaps the grab UV to the nearest real mesh point and converts xyz to world.
"""
import os, sys, json, re, argparse, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="overhead UV-overlay PNG from head_RL.py")
ap.add_argument("--out",   required=True, help="output JSON path")
args = ap.parse_args()

img_path = os.path.abspath(args.image)
if not os.path.exists(img_path):
    print(f"[vlm_action] ERROR: image not found: {img_path}")
    sys.exit(1)

PROMPT = f"""Read the image file at {img_path}

A SINGLE robot arm must flatten the garment on a table. The image is the real overhead camera view
of the crumpled garment, plus a SMALL reference window on the right.

MAIN image (most of the picture) — the overhead camera. Every point on the fabric is a small dot
coloured by its predicted UV. Raised, bunched, folded fabric (you can see it from the shading and
structure) is what needs to be grabbed and spread out. A WHITE OUTLINE is drawn on the FLOOR behind
the garment: that is the TARGET — the exact shape AND orientation the garment must end up in when
flattened (body + sleeves in a fixed pose). Drag fabric so the garment fills this white outline:
fabric sticking out past the outline, or area inside the outline left empty, must be fixed. The
outline does NOT move or rotate — turn and spread the garment to match it.

SMALL reference window (top right, "flat UV / colour key") — the garment laid out perfectly flat,
coloured by UV. Horizontal axis = u (0..1, left→right), vertical axis = v (0..1, bottom→top). Use it
only as a colour key: a dot's colour on the MAIN image tells you which part of the flat garment that
fabric is, so you know where inside the white outline it belongs.

UV colour code (BOTH panels): each dot's colour encodes its UV coordinate — Red channel = u,
Green channel = v. On the RIGHT this colour matches position (it IS the flat layout). On the LEFT,
each crumpled point is painted with its predicted UV colour. So find a coloured region on the LEFT,
read its colour, and the same colour on the RIGHT template tells you where that fabric belongs when
flat. Use this to choose which crumpled region to grab and where to drag it.

WORLD FRAME for all x,y,z below (metres): origin = the garment's centre on the table; +x = right,
+y = away from the camera, +z = up. The table surface is z = 0. The garment spans roughly
x,y in -0.3..0.3.

Task — pick ONE grab + drag:
  1. grab_u, grab_v : the UV (colour) of a RAISED / BUNCHED region on the LEFT you want to pick up.
  2. release [x,y,z]: where to carry that fabric and let go — the spot inside the WHITE target
     outline where its (u,v) belongs (use the RIGHT panel colour key), laid down flat
     (z ~= 0.0-0.03, i.e. on the table).
  3. path [5 points]: 5 waypoints [x,y,z] the hand moves through between grab and release. Lift the
     fabric up first (z ~= 0.15-0.30) so it clears the table, carry it across, then come down to the
     release. The 5 points should go from near the grab, up and over, down to near the release.

Bounds: x,y in -0.5..0.5 ; z in 0.0..0.40.

Respond with ONLY valid JSON, no markdown. Use plain numbers — NO leading '+' sign
(write 0.15, not +0.15) and no units:
{{
  "grab_u": 0.XX, "grab_v": 0.XX,
  "release": [0.XX, 0.XX, 0.XX],
  "path": [[0.XX,0.XX,0.XX],[0.XX,0.XX,0.XX],[0.XX,0.XX,0.XX],[0.XX,0.XX,0.XX],[0.XX,0.XX,0.XX]],
  "reasoning": "one sentence"
}}"""

result = subprocess.run(
    ["claude", "-p", PROMPT, "--model", "claude-haiku-4-5", "--allowedTools", "Read"],
    capture_output=True, text=True,
)

raw = result.stdout.strip()
if not raw:
    print(f"[vlm_action] ERROR: no output from claude CLI\n{result.stderr}")
    sys.exit(1)

# Haiku often wraps the JSON in prose and/or a ```json fence. Extract the JSON object:
# prefer a fenced block, else fall back to the outermost {...} span.
m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
if m:
    raw = m.group(1)
else:
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b != -1 and b > a:
        raw = raw[a:b+1]
raw = raw.strip()

# sanitize: strip leading '+' on numbers (JSON forbids them) — only before a digit
# that follows a delimiter, so '+' inside reasoning strings is left alone
raw = re.sub(r'([:\[,]\s*)\+(\d)', r'\1\2', raw)

PATH_LEN = 5
_clampxy  = lambda c: float(max(-0.5, min(0.5, c)))
_clampz   = lambda c: float(max(0.0,  min(0.40, c)))


def _xyz(p):
    return [_clampxy(p[0]), _clampxy(p[1]), _clampz(p[2])]


def _fix_path(path, release):
    """Coerce to exactly PATH_LEN clamped [x,y,z] waypoints (pad with release / truncate)."""
    pts = [_xyz(p) for p in (path or []) if isinstance(p, (list, tuple)) and len(p) >= 3]
    pts = pts[:PATH_LEN]
    while len(pts) < PATH_LEN:
        pts.append(list(release))
    return pts


try:
    data = json.loads(raw)

    grab_u  = float(max(0.0, min(1.0, data["grab_u"])))
    grab_v  = float(max(0.0, min(1.0, data["grab_v"])))
    release = _xyz(data["release"])
    path    = _fix_path(data.get("path"), release)
    result_data = {"arm1": {
        "grab_u": grab_u, "grab_v": grab_v,
        "release": release, "path": path,
        "reasoning": str(data.get("reasoning", "")),
    }}

except (json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError) as e:
    print(f"[vlm_action] WARNING: parse failed ({e}): {raw!r}")
    rel = [0.20, 0.10, 0.02]
    result_data = {"arm1": {
        "grab_u": 0.5, "grab_v": 0.5,
        "release": rel,
        "path": [[0.0, 0.0, 0.20], [0.10, 0.05, 0.25],
                 [0.15, 0.08, 0.20], [0.18, 0.09, 0.10], list(rel)],
        "reasoning": "fallback",
    }}

with open(args.out, "w") as fh:
    json.dump(result_data, fh, indent=2)

for arm, a in result_data.items():
    r = a["release"]
    print(f"[vlm_action] {arm} grab_uv=({a['grab_u']:.2f},{a['grab_v']:.2f})"
          f"  release=({r[0]:.2f},{r[1]:.2f},{r[2]:.2f})  +{len(a['path'])} waypoints"
          f"  → {a['reasoning']}")
