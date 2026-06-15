"""
label_uv_groups.py — interactive UV-space lasso tool for building grasp_config.yaml.

Env: fold  (matplotlib + numpy only, no spconv needed)

Load a partial or full npz, lasso regions in UV space, name each group,
and get the YAML snippet to paste into grasp_config.yaml.

Usage:
    conda run -n fold python data/label_uv_groups.py data/majca_test/partial/majca_0000.npz
    conda run -n fold python data/label_uv_groups.py data/majca_test/full/majca_0000.npz

Controls:
    Click + drag      — draw a lasso around UV points
    Enter (terminal)  — confirm group name / settings after each lasso
    d                 — delete last group
    q / close window  — finish and print YAML
"""

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── colour palette for groups ─────────────────────────────────────────────────────────────
_COLOURS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#8bc34a"]


def _load_uv(npz_path):
    d = np.load(npz_path)
    if "panel_uv" in d.files:
        uv       = d["panel_uv"].astype(np.float32)   # (N, 2)
        panel_id = d["panel_id"].astype(np.int32)      # (N,)
    else:
        raise KeyError(f"{npz_path} has no 'panel_uv' key — use a partial or full npz from data_gen capture")
    return uv, panel_id


def _fit_circle(uv_pts):
    """Fit a circle to a set of UV points: returns (center, radius)."""
    c = uv_pts.mean(0)
    r = float(np.linalg.norm(uv_pts - c, axis=1).max())
    return c, r


def _yaml_snippet(groups):
    lines = ["grasp_groups:\n"]
    for g in groups:
        c, r = g["center"], g["radius"]
        lines.append(f"  - name: {g['name']}")
        lines.append(f"    panel: {g['panel']}")
        lines.append(f"    uv_center: [{c[0]:.4f}, {c[1]:.4f}]")
        lines.append(f"    uv_radius: {r:.4f}")
        lines.append("")
    return "\n".join(lines)


def run(npz_path):
    uv, panel_id = _load_uv(npz_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("UV Lasso Tool — lasso a region, then fill in group details in the terminal\n"
                 "  [d] = delete last group    [q / close] = finish & print YAML", fontsize=10)

    panel_names = {0: "Front (panel 0)", 1: "Back (panel 1)"}
    scatters    = {}
    for ax, pid in zip(axes, [0, 1]):
        mask = panel_id == pid
        pts  = uv[mask]
        sc   = ax.scatter(pts[:, 0], pts[:, 1], s=4, c="#aaaaaa", alpha=0.5, picker=False)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("U"); ax.set_ylabel("V")
        ax.set_title(panel_names[pid])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        scatters[pid] = (ax, sc, mask, pts)

    groups    = []          # list of dicts
    artists   = []          # (ax, patch, text) for each group
    _state    = {"active_pid": None, "lasso": None}

    legend_ax = fig.add_axes([0.01, 0.01, 0.98, 0.04])
    legend_ax.axis("off")
    _legend_text = legend_ax.text(0.5, 0.5, "No groups yet", ha="center", va="center",
                                  fontsize=8, transform=legend_ax.transAxes)

    def _refresh_legend():
        if not groups:
            _legend_text.set_text("No groups yet")
        else:
            parts = [f"[{i+1}] {g['name']}  uv=({g['center'][0]:.3f},{g['center'][1]:.3f})  r={g['radius']:.3f}  panel={g['panel']}"
                     for i, g in enumerate(groups)]
            _legend_text.set_text("   |   ".join(parts))
        fig.canvas.draw_idle()

    def _on_lasso_select(verts, pid):
        path  = Path(verts)
        _, _, mask, pts = scatters[pid]
        inside = path.contains_points(pts)
        sel_uv = pts[inside]
        if len(sel_uv) < 3:
            print(f"  [label_uv] < 3 points selected on panel {pid} — try again")
            return

        c, r = _fit_circle(sel_uv)
        colour = _COLOURS[len(groups) % len(_COLOURS)]

        # draw circle + label on the axis
        ax = scatters[pid][0]
        patch = mpatches.Circle(c, r, fill=False, edgecolor=colour, linewidth=2, linestyle="--")
        ax.add_patch(patch)
        # highlight selected points
        hi = ax.scatter(sel_uv[:, 0], sel_uv[:, 1], s=10, c=colour, alpha=0.8, zorder=5)
        gid = len(groups) + 1
        txt = ax.text(c[0], c[1], str(gid), color=colour, fontsize=12, fontweight="bold",
                      ha="center", va="center", zorder=6)
        fig.canvas.draw_idle()

        # ask user for group details in terminal
        print(f"\n── Group {gid} ────────────────────────────────")
        print(f"  Panel         : {pid}  ({panel_names[pid]})")
        print(f"  Points selected: {inside.sum()}")
        print(f"  UV center     : ({c[0]:.4f}, {c[1]:.4f})")
        print(f"  UV radius     : {r:.4f}")
        name = input("  Name (e.g. front_collar): ").strip() or f"group_{gid}"

        groups.append({
            "name":   name,
            "panel":  pid,
            "center": c,
            "radius": round(r + 0.01, 4),
            "n_pts":  int(inside.sum()),
        })
        artists.append((ax, patch, hi, txt))
        _refresh_legend()
        print(f"  → saved as group {gid}: '{name}'")

    # attach both lassos upfront — one per panel, always active
    for pid in (0, 1):
        ax = scatters[pid][0]
        _lasso = LassoSelector(ax, lambda v, p=pid: _on_lasso_select(v, p), useblit=True, button=1)
        _state[f"lasso_{pid}"] = _lasso   # keep reference so GC doesn't collect it

    def _on_key(event):
        if event.key == "d":
            if not groups:
                print("  [label_uv] no groups to delete")
                return
            groups.pop()
            ax, patch, hi, txt = artists.pop()
            patch.remove(); hi.remove(); txt.remove()
            print(f"  [label_uv] deleted last group ({len(groups)} remain)")
            _refresh_legend()
            fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    print("\n[label_uv_groups] Click and drag on a panel to lasso a UV region.")
    print("  Press 'd' to delete the last group, 'q' or close window to finish.\n")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()

    # ── output ───────────────────────────────────────────────────────────────────────────
    if not groups:
        print("[label_uv_groups] No groups defined.")
        return

    print("\n" + "═" * 60)
    print("YAML SNIPPET — paste into grasp_config.yaml:")
    print("═" * 60)
    snippet = _yaml_snippet(groups)
    print(snippet)

    out = os.path.join(_ROOT, "grasp_groups_export.yaml")
    with open(out, "w") as f:
        f.write(snippet)
    print(f"[label_uv_groups] also saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="partial or full npz with panel_uv + panel_id keys")
    args = ap.parse_args()
    run(args.npz)
