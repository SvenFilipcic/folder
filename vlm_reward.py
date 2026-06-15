"""
vlm_reward.py — VLM flatness scoring via Claude CLI.

Called by head_RL.py after each drag-settle cycle:
    python vlm_reward.py --image rl_reward_img.png --out /tmp/vlm_score.json

Score [0,1]: 0=deeply crumpled, 1=perfectly flat.
"""
import os, sys, json, argparse, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="overhead RGB PNG path")
ap.add_argument("--out",   required=True, help="output JSON path")
args = ap.parse_args()

img_path = os.path.abspath(args.image)
if not os.path.exists(img_path):
    print(f"[vlm_reward] ERROR: image not found: {img_path}")
    sys.exit(1)

PROMPT = f"""Read the image file at {img_path}

Rate how flat this garment is on a scale from 0.0 to 1.0:
  0.0 — deeply crumpled, bunched into a pile
  0.3 — significant folds, clearly not flat
  0.5 — partially unfolded, some wrinkles remain
  0.7 — mostly flat with minor folds at edges
  1.0 — perfectly flat, evenly spread

Respond with ONLY valid JSON, nothing else:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""

result = subprocess.run(
    ["claude", "-p", PROMPT, "--model", "claude-haiku-4-5", "--allowedTools", "Read"],
    capture_output=True, text=True,
)

raw = result.stdout.strip()
if not raw:
    print(f"[vlm_reward] ERROR: no output\n{result.stderr}")
    sys.exit(1)

try:
    parsed = json.loads(raw)
    score  = float(max(0.0, min(1.0, parsed["score"])))
    result_data = {"score": score, "reasoning": parsed.get("reasoning", "")}
except (json.JSONDecodeError, KeyError, ValueError) as e:
    print(f"[vlm_reward] WARNING: parse failed ({e}): {raw!r}")
    result_data = {"score": 0.0, "reasoning": f"parse_error: {raw}"}

with open(args.out, "w") as fh:
    json.dump(result_data, fh)

print(f"[vlm_reward] score={result_data['score']:.3f}  {result_data['reasoning']!r}")
