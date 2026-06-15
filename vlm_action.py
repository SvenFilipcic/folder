"""
vlm_action.py — VLA: overhead UV-overlay image → 2-arm grab+drag actions via Claude Haiku.

Called by head_RL.py in VLA mode:
    python vlm_action.py --image rl_uv_overlay.png --out /tmp/vlm_action.json

head_RL.py renders an overhead view of the garment where every point is coloured by its
predicted UV (R=u, G=v) and a sparse set of points is labelled with the literal (u,v).
Haiku reads that image and picks, per arm, a grab in UV space + a world-space drag.
head_RL.py then snaps each grab UV to the nearest real mesh point.
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

The image has TWO panels of the SAME garment, which two robot arms must flatten on a table.

LEFT panel — the real overhead camera view of the crumpled garment. Every point on the fabric is
drawn as a small dot. Raised, bunched, folded fabric (you can see it from the shading/structure)
is what needs to be grabbed and spread out.

RIGHT panel — the garment's TRUE flat layout ("UV space"): a fixed reference template showing the
garment laid out perfectly flat (you will see its real shape — body and sleeves). The horizontal
axis is u (0..1, left→right), the vertical axis is v (0..1, bottom→top).

UV colour code (BOTH panels): each dot's colour encodes its UV coordinate — Red channel = u,
Green channel = v. On the RIGHT this colour matches position (it IS the flat layout). On the LEFT,
each crumpled point is painted with its predicted UV colour. So find a coloured region on the LEFT,
read its colour, and the same colour on the RIGHT template tells you where that fabric belongs when
flat. Use this to choose which crumpled region to grab and which direction to drag it.

Task: pick ONE grab point per arm, specified by the UV coordinate (grab_u, grab_v) you want to
grab, plus a drag vector. Prefer grabbing raised/bunched regions (LEFT) and dragging them outward
toward where their (u, v) says they belong (RIGHT). The two grabs are enforced >=5 cm apart in 3D.

Drag vector (world metres, LEFT panel axes — right = +x, away from camera = +y):
  dx: -0.5..0.5 (right+)
  dy: -0.5..0.5 (away+)
  dz:  0.05..0.40 lift height during the drag (use 0.15-0.25)

Respond with ONLY valid JSON, no markdown. Use plain numbers — NO leading '+' sign
(write 0.15, not +0.15) and no units:
{{
  "arm1": {{"grab_u": 0.XX, "grab_v": 0.XX, "dx": 0.XX, "dy": 0.XX, "dz": 0.XX, "reasoning": "one sentence"}},
  "arm2": {{"grab_u": 0.XX, "grab_v": 0.XX, "dx": 0.XX, "dy": 0.XX, "dz": 0.XX, "reasoning": "one sentence"}}
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

try:
    data = json.loads(raw)

    def _parse_arm(d):
        return {
            "grab_u": float(max(0.0, min(1.0, d["grab_u"]))),
            "grab_v": float(max(0.0, min(1.0, d["grab_v"]))),
            "dx":     float(max(-0.5, min(0.5, d["dx"]))),
            "dy":     float(max(-0.5, min(0.5, d["dy"]))),
            "dz":     float(max(0.05, min(0.40, d["dz"]))),
            "reasoning": str(d.get("reasoning", "")),
        }

    result_data = {"arm1": _parse_arm(data["arm1"]), "arm2": _parse_arm(data["arm2"])}

except (json.JSONDecodeError, KeyError, ValueError) as e:
    print(f"[vlm_action] WARNING: parse failed ({e}): {raw!r}")
    result_data = {
        "arm1": {"grab_u": 0.25, "grab_v": 0.25, "dx":  0.15, "dy":  0.10, "dz": 0.20, "reasoning": "fallback"},
        "arm2": {"grab_u": 0.75, "grab_v": 0.75, "dx": -0.15, "dy": -0.10, "dz": 0.20, "reasoning": "fallback"},
    }

with open(args.out, "w") as fh:
    json.dump(result_data, fh, indent=2)

for arm in ("arm1", "arm2"):
    a = result_data[arm]
    print(f"[vlm_action] {arm} grab_uv=({a['grab_u']:.2f},{a['grab_v']:.2f})"
          f"  drag=({a['dx']:.2f},{a['dy']:.2f},{a['dz']:.2f})  → {a['reasoning']}")
