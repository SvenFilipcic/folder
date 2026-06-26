"""
label_grasp_regions.py — interactive tool to define the ORDERED grab regions the deterministic UV
teacher replays (`head_RL.py --scripted`). Env: fold (matplotlib + numpy only).

You lasso patches on the CANONICAL flat garment (its UV layout). Each lasso, in the order you draw
it, becomes one priority region: the teacher will grab that UV patch and drag it to the flat spot
shown, then move to the next. Think "what would I flatten first" — e.g. pin the body, then pull each
sleeve out, then the hem.

    conda run -n fold python data/label_grasp_regions.py            # annotate on the reference garment
    conda run -n fold python data/label_grasp_regions.py --out reference/grasp_regions.json

Controls:
    Click + drag      — lasso a UV region (on either panel); you'll name it in the terminal
    d                 — delete the last region
    q / close window  — finish & save reference/grasp_regions.json (regions kept IN DRAW ORDER)

Output: reference/grasp_regions.json  — list of {name, uv_center, uv_radius, panel, target_off},
where target_off is the centroid-relative flat target of the actually-selected verts.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import grasp_regions as gr

_COLOURS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#8bc34a"]
_PANEL_NAMES = {0: "panel 0 (front)", 1: "panel 1 (back)"}


def run(out_path):
    node_uv, node_panel, flat_pts = gr.load_reference()
    flat_cen = flat_pts[:, :2].mean(0)

    panels = sorted(np.unique(node_panel).tolist())
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6), squeeze=False)
    axes = axes[0]
    fig.suptitle("Grasp-region tool — lasso patches IN THE ORDER you want them flattened\n"
                 "[d] delete last   [q / close] finish & save", fontsize=10)

    scatters = {}
    for ax, pid in zip(axes, panels):
        gidx = np.where(node_panel == pid)[0]          # global vertex indices on this panel
        pts  = node_uv[gidx]
        ax.scatter(pts[:, 0], pts[:, 1], s=4, c="#aaaaaa", alpha=0.5)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("u"); ax.set_ylabel("v"); ax.set_aspect("equal")
        ax.set_title(_PANEL_NAMES.get(pid, f"panel {pid}")); ax.grid(True, alpha=0.3)
        scatters[pid] = (ax, gidx, pts)

    regions = []     # ordered region dicts (draw order = priority)
    artists = []     # (patch, hi, txt) per region for undo
    keep    = {}     # keep lasso refs alive

    legend = fig.add_axes([0.01, 0.01, 0.98, 0.04]); legend.axis("off")
    leg_txt = legend.text(0.5, 0.5, "No regions yet", ha="center", va="center", fontsize=8)

    def _refresh():
        if regions:
            leg_txt.set_text("   |   ".join(
                f"[{i+1}] {r['name']} uv=({r['uv_center'][0]:.2f},{r['uv_center'][1]:.2f}) "
                f"r={r['uv_radius']:.2f}" for i, r in enumerate(regions)))
        else:
            leg_txt.set_text("No regions yet")
        fig.canvas.draw_idle()

    def _on_lasso(verts, pid):
        ax, gidx, pts = scatters[pid]
        inside = Path(verts).contains_points(pts)
        if inside.sum() < 3:
            print(f"  [regions] < 3 points on {_PANEL_NAMES.get(pid, pid)} — try again"); return
        sel_glob = gidx[inside]                          # selected global vertex indices
        uc  = node_uv[sel_glob].mean(0)
        rad = float(np.linalg.norm(node_uv[sel_glob] - uc, axis=1).max()) + 0.01
        off = flat_pts[sel_glob, :2].mean(0) - flat_cen  # centroid-relative flat target

        colour = _COLOURS[len(regions) % len(_COLOURS)]
        patch  = mpatches.Circle(uc, rad, fill=False, edgecolor=colour, lw=2, ls="--"); ax.add_patch(patch)
        hi     = ax.scatter(node_uv[sel_glob, 0], node_uv[sel_glob, 1], s=10, c=colour, zorder=5)
        gid    = len(regions) + 1
        txt    = ax.text(uc[0], uc[1], str(gid), color=colour, fontsize=12, fontweight="bold",
                         ha="center", va="center", zorder=6)
        fig.canvas.draw_idle()

        print(f"\n── Region {gid} ── {_PANEL_NAMES.get(pid, pid)} | {inside.sum()} pts | "
              f"uv=({uc[0]:.3f},{uc[1]:.3f}) r={rad:.3f} | target_off=({off[0]:.3f},{off[1]:.3f})")
        name = input("  name (e.g. body_center / lsleeve_tip / hem_left): ").strip() or f"region_{gid}"
        regions.append({
            "name": name, "uv_center": [float(uc[0]), float(uc[1])], "uv_radius": rad,
            "panel": int(pid),
            "target_off": [float(off[0]), float(off[1]), float(flat_pts[:, 2].mean())],
        })
        artists.append((patch, hi, txt)); _refresh()
        print(f"  → region {gid} '{name}' saved")

    for pid in panels:
        ax = scatters[pid][0]
        keep[pid] = LassoSelector(ax, lambda v, p=pid: _on_lasso(v, p), useblit=True, button=1)

    def _on_key(ev):
        if ev.key == "d" and regions:
            regions.pop()
            patch, hi, txt = artists.pop(); patch.remove(); hi.remove(); txt.remove()
            print(f"  [regions] deleted last ({len(regions)} remain)"); _refresh()
        elif ev.key == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    print("\n[label_grasp_regions] lasso regions in priority order; 'd' undo, 'q'/close to save.\n")
    plt.tight_layout(rect=[0, 0.06, 1, 1]); plt.show()

    if not regions:
        print("[label_grasp_regions] no regions defined — nothing saved."); return
    gr.save(regions, out_path)
    print(f"\n[label_grasp_regions] saved {len(regions)} regions (priority order) → {out_path}")
    for i, r in enumerate(regions):
        print(f"  {i+1}. {r['name']:<16} uv=({r['uv_center'][0]:.2f},{r['uv_center'][1]:.2f}) "
              f"r={r['uv_radius']:.2f}  target_off=({r['target_off'][0]:.2f},{r['target_off'][1]:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=gr.DEFAULT_PATH, help="output JSON (default reference/grasp_regions.json)")
    run(ap.parse_args().out)
