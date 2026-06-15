"""
infer_franka.py  —  Full garment → grasp pipeline orchestrator.

Workflow:
  1. Generate a crumpled garment npz via data_gen.py (fold conda, Isaac Sim)
     OR point at an existing npz with --npz.
  2. Run UV Mapper inference + priority-group grasp selection (infer_grasp.py, infer env).
  3. Publish the grasp point to the Franka via ROS2 (ros2/grasp_publisher.py).

Usage:
    # generate + infer + publish (full pipeline):
    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python infer_franka.py --gui

    # skip generation, run inference on an existing npz:
    conda run -n infer python infer_franka.py --npz data/majca/partial/majca_0000.npz

    # generate + infer, print result only (no ROS):
    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python infer_franka.py --no-ros

    # with camera→robot extrinsic for real lab:
    conda run -n infer python infer_franka.py --npz majca_0000.npz --tf "0.5 0.0 0.4 0 0 0 1"
"""

import os, sys, json, argparse, subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description="garment → UV inference → Franka grasp pipeline")

    # ── generation ───────────────────────────────────────────────────────────────────────
    gen = ap.add_argument_group("generation (skipped if --npz is given)")
    gen.add_argument("--npz",         default=None,
                     help="use this existing partial npz instead of running data_gen")
    gen.add_argument("--gui",         action="store_true", help="open Isaac GUI during generation")
    gen.add_argument("--ball",        action="store_true", default=True,
                     help="crumple with ball drape (default: on)")
    gen.add_argument("--no-ball",     dest="ball", action="store_false")
    gen.add_argument("--outdir",      default="data/majca",
                     help="dataset dir for generated npz (partial/ + full/ subdirs)")
    gen.add_argument("--prefix",      default="majca", help="npz filename prefix")
    gen.add_argument("--drape-frames",type=int, default=60)
    gen.add_argument("--settle",      type=int, default=80)

    # ── inference ────────────────────────────────────────────────────────────────────────
    inf = ap.add_argument_group("inference")
    inf.add_argument("--config",  default=None,
                     help="grasp_config.yaml path (default: <root>/grasp_config.yaml)")
    inf.add_argument("--model",   default=None,
                     help="UV Mapper checkpoint (default: checkpoints/uv_mapper_best.pth)")
    inf.add_argument("--device",  default=None, help="cuda / cpu (default: auto)")
    inf.add_argument("--infer-env", default="infer",
                     help="conda env that has spconv (default: infer)")
    inf.add_argument("--grasp-out", default="grasp_result.json",
                     help="where to write the grasp JSON result")

    # ── publishing ───────────────────────────────────────────────────────────────────────
    pub = ap.add_argument_group("publishing")
    pub.add_argument("--no-ros",  action="store_true",
                     help="skip ROS2 publishing (print result only)")
    pub.add_argument("--tf",      default=None,
                     help="camera→robot_base extrinsic: 'tx ty tz qx qy qz qw'")
    pub.add_argument("--watch",   action="store_true",
                     help="keep the ROS publisher alive, re-publish on each new json")

    return ap.parse_args()


# ── step 1: generate npz via data_gen.py (fold conda, Isaac Sim) ─────────────────────────
def _run_generation(args):
    """Launch data_gen.py in the current environment (must already be fold conda + Isaac)."""
    data_gen = os.path.join(_ROOT, "data", "data_gen.py")
    outdir   = args.outdir if os.path.isabs(args.outdir) else os.path.join(_ROOT, args.outdir)
    part_dir = os.path.join(outdir, "partial")

    cmd = [sys.executable, data_gen,
           "--outdir", outdir,
           "--prefix", args.prefix,
           "--drape-frames", str(args.drape_frames),
           "--settle", str(args.settle),
           "--samples", "1"]
    if args.gui:  cmd.append("--gui")
    if args.ball: cmd.append("--ball")

    print(f"[pipeline] generating garment sample via data_gen.py ...")
    ret = subprocess.run(cmd, cwd=_ROOT)
    if ret.returncode != 0:
        print("[pipeline] ✗ data_gen.py failed")
        sys.exit(1)

    existing = sorted(
        f for f in os.listdir(part_dir)
        if f.startswith(args.prefix) and f.endswith(".npz")
    )
    if not existing:
        print(f"[pipeline] ✗ no npz found in {part_dir} after generation")
        sys.exit(1)
    npz = os.path.join(part_dir, existing[-1])
    print(f"[pipeline] generated → {npz}")
    return npz


# ── step 2: run inference in the infer conda env ─────────────────────────────────────────
def _run_inference(npz_path, args):
    """Call infer_grasp.py via conda run in the infer env, returns result dict or None."""
    infer_script = os.path.join(_ROOT, "infer_grasp.py")
    out_path     = args.grasp_out if os.path.isabs(args.grasp_out) \
                   else os.path.join(_ROOT, args.grasp_out)
    cfg_path     = args.config  or os.path.join(_ROOT, "grasp_config.yaml")
    mdl_path     = args.model   or os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")

    cmd = ["conda", "run", "-n", args.infer_env, "--no-capture-output",
           "python", infer_script, npz_path,
           "--config", cfg_path,
           "--model",  mdl_path,
           "--out",    out_path]
    if args.device:
        cmd += ["--device", args.device]

    print(f"[pipeline] running UV Mapper inference (env: {args.infer_env}) ...")
    ret = subprocess.run(cmd, cwd=_ROOT)
    if ret.returncode != 0:
        print("[pipeline] ✗ inference failed — check infer env / model checkpoint")
        return None

    if not os.path.exists(out_path):
        print(f"[pipeline] ✗ inference produced no output at {out_path}")
        return None

    with open(out_path) as f:
        result = json.load(f)
    return result if result.get("found") else None


# ── step 3: publish via ROS2 ─────────────────────────────────────────────────────────────
def _run_publisher(result, args):
    """Try to publish via rclpy directly; fall back to dry-run print."""
    try:
        import rclpy
        from rclpy.node import Node
        _has_ros = True
    except ImportError:
        _has_ros = False

    tf = [float(v) for v in args.tf.split()] if args.tf else None

    if not _has_ros:
        sys.path.insert(0, _ROOT)
        from ros2.grasp_publisher import publish_grasp
        print("[pipeline] rclpy not found — dry-run print:")
        publish_grasp(result, tf, node=None)
        return

    sys.path.insert(0, _ROOT)
    from ros2.grasp_publisher import GraspPublisherNode, publish_grasp
    import rclpy

    out_path = args.grasp_out if os.path.isabs(args.grasp_out) \
               else os.path.join(_ROOT, args.grasp_out)

    rclpy.init()
    node = GraspPublisherNode(out_path, tf, watch=args.watch)
    if args.watch:
        print("[pipeline] ROS publisher in --watch mode (Ctrl-C to stop)")
        rclpy.spin(node)
    else:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


# ── main ─────────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. get the npz
    if args.npz:
        npz_path = args.npz if os.path.isabs(args.npz) else os.path.join(_ROOT, args.npz)
        if not os.path.exists(npz_path):
            print(f"[pipeline] ✗ --npz file not found: {npz_path}")
            sys.exit(1)
        print(f"[pipeline] using existing npz: {npz_path}")
    else:
        npz_path = _run_generation(args)

    # 2. inference
    result = _run_inference(npz_path, args)
    if result is None:
        print("[pipeline] no grasp point found — exiting")
        sys.exit(0)

    print(f"\n[pipeline] GRASP POINT SELECTED:")
    import numpy as np
    print(f"  group      : {result['group_id']} — {result['group_name']}")
    print(f"  xyz_world  : {np.array(result['xyz_world']).round(4)}")
    print(f"  normal     : {np.array(result['normal']).round(4)}")
    print(f"  uv         : {np.array(result['uv']).round(4)}")
    print(f"  confidence : {result['confidence']:.3f}  (n_matching={result['n_matching']})")

    # 3. publish
    if args.no_ros:
        print("[pipeline] --no-ros set, skipping ROS2 publish")
        return

    _run_publisher(result, args)


if __name__ == "__main__":
    main()
