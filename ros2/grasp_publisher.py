"""
grasp_publisher.py — ROS2 Humble node: publishes a grasp target to the Franka.

Reads grasp_result.json written by infer_grasp.py, publishes:
  /grasp_target   geometry_msgs/PoseStamped  — position + approach orientation
  /grasp_marker   visualization_msgs/Marker  — sphere for RViz

The orientation encodes the approach direction: gripper Z-axis aligns with -normal
(approach from above along the surface normal direction).

Coordinate note:
  In sim:  xyz_world is already in robot world frame (Isaac world = table frame).
  In lab:  xyz_world is in CAMERA frame. You must apply the camera→robot_base
           extrinsic transform (from hand-eye calibration) before publishing.
           Set CAMERA_TO_BASE_TF in the env or pass --tf "x y z qx qy qz qw".

Usage:
    source /opt/ros/humble/setup.bash
    python ros2/grasp_publisher.py grasp_result.json
    python ros2/grasp_publisher.py grasp_result.json --tf "0.5 0.0 0.4 0 0 0 1"
    python ros2/grasp_publisher.py --watch  # re-publish whenever grasp_result.json changes
"""

import os, sys, json, argparse, time
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from visualization_msgs.msg import Marker
    from std_msgs.msg import Header
    _ROS = True
except ImportError:
    _ROS = False
    print("[grasp_publisher] WARNING: rclpy not found — dry-run mode (prints only)")


def _normal_to_quat(normal):
    """
    Quaternion that rotates world Z to align with -normal (approach along normal).
    Returns (qx, qy, qz, qw).
    """
    n  = np.array(normal, dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-9
    approach = -n                              # gripper approaches FROM above along -normal
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, approach)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, approach))
    if sin_a < 1e-6:
        return (0.0, 0.0, 0.0, 1.0) if cos_a > 0 else (1.0, 0.0, 0.0, 0.0)
    axis /= sin_a
    half  = np.arctan2(sin_a, cos_a) / 2.0
    return (*( axis * np.sin(half)), float(np.cos(half)))   # qx, qy, qz, qw


def _apply_tf(xyz, tf):
    """Apply a 7-vector extrinsic [tx ty tz qx qy qz qw] to xyz."""
    if tf is None:
        return xyz
    t = np.array(tf[:3]); q = np.array(tf[3:])   # qx qy qz qw
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),  2*(x*z+w*y)],
        [2*(x*y+w*z),  1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),  1-2*(x*x+y*y)],
    ])
    return (R @ np.array(xyz)) + t


def publish_grasp(result, tf=None, node=None):
    """Publish one grasp result. node=None → dry-run print."""
    xyz    = _apply_tf(result["xyz_world"], tf)
    normal = np.array(result["normal"], dtype=np.float64)
    qx, qy, qz, qw = _normal_to_quat(normal)

    print(f"[grasp_publisher] publishing group {result['group_id']} '{result['group_name']}'")
    print(f"  xyz  (robot frame) : {np.array(xyz).round(4)}")
    print(f"  normal             : {normal.round(4)}")
    print(f"  orientation (xyzw) : [{qx:.4f} {qy:.4f} {qz:.4f} {qw:.4f}]")
    print(f"  confidence         : {result['confidence']:.3f}  uv={np.array(result['uv']).round(3)}")

    if node is None:
        return

    header = Header()
    header.stamp    = node.get_clock().now().to_msg()
    header.frame_id = "panda_link0"   # Franka base frame

    pose_msg                 = PoseStamped()
    pose_msg.header          = header
    pose_msg.pose.position.x = float(xyz[0])
    pose_msg.pose.position.y = float(xyz[1])
    pose_msg.pose.position.z = float(xyz[2])
    pose_msg.pose.orientation.x = qx
    pose_msg.pose.orientation.y = qy
    pose_msg.pose.orientation.z = qz
    pose_msg.pose.orientation.w = qw
    node.pose_pub.publish(pose_msg)

    marker              = Marker()
    marker.header       = header
    marker.type         = Marker.SPHERE
    marker.action       = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 0.03
    marker.color.r = 1.0; marker.color.g = 0.4; marker.color.b = 0.0; marker.color.a = 1.0
    marker.pose         = pose_msg.pose
    node.marker_pub.publish(marker)


class GraspPublisherNode(Node):
    def __init__(self, json_path, tf, watch):
        super().__init__("grasp_publisher")
        self.json_path  = json_path
        self.tf         = tf
        self.watch      = watch
        self.last_mtime = None
        self.pose_pub   = self.create_publisher(PoseStamped, "/grasp_target", 10)
        self.marker_pub = self.create_publisher(Marker,      "/grasp_marker", 10)
        if watch:
            self.create_timer(0.5, self._check_file)
        else:
            self._publish_file()

    def _publish_file(self):
        with open(self.json_path) as f:
            result = json.load(f)
        if not result.get("found"):
            self.get_logger().warn("grasp_result.json has found=false — nothing to publish")
            return
        publish_grasp(result, self.tf, self)

    def _check_file(self):
        try:
            mtime = os.path.getmtime(self.json_path)
        except FileNotFoundError:
            return
        if mtime != self.last_mtime:
            self.last_mtime = mtime
            self._publish_file()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json",   nargs="?", default="grasp_result.json",
                    help="path to grasp_result.json from infer_grasp.py")
    ap.add_argument("--tf",   default=None,
                    help="camera→robot_base extrinsic: 'tx ty tz qx qy qz qw'")
    ap.add_argument("--watch", action="store_true",
                    help="re-publish whenever the json file changes (polling)")
    args = ap.parse_args()

    tf = [float(v) for v in args.tf.split()] if args.tf else None

    if not _ROS:
        with open(args.json) as f:
            result = json.load(f)
        if result.get("found"):
            publish_grasp(result, tf, node=None)
        else:
            print("[grasp_publisher] found=false — nothing to publish")
        return

    rclpy.init()
    node = GraspPublisherNode(args.json, tf, args.watch)
    if args.watch:
        rclpy.spin(node)
    else:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
