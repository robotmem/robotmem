#!/usr/bin/env python3
"""robotmem ROS 2 Demo — 记忆驱动导航对比

Session 1: 反应式避障探索，robotmem 记住路线
Session 2: recall 航点 → 直线导航

证明：走过的路不用再走第二次。

环境要求：
- ROS 2 Humble
- Originbot 仿真（The Construct 平台）或任何有 /xxx/cmd_vel + /xxx/odom 的移动机器人
- pip install robotmem

运行：
  source /opt/ros/humble/setup.bash
  python3 demo.py              # 默认 25 秒探索
  python3 demo.py 40           # 40 秒探索
  python3 demo.py 25 --prefix /turtlebot3  # 自定义 topic 前缀
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math, time, json, os, sys, shutil, argparse

from robotmem.sdk import RobotMemory

DB_DIR = os.path.expanduser("~/.robotmem-ros-demo")
DB_PATH = os.path.join(DB_DIR, "memory.db")
COLLECTION = "ros_nav_demo"


class ExploreNode(Node):
    """Session 1: 前进 + 碰墙转弯，记录航点"""

    def __init__(self, duration=25.0, cmd_topic="/originbot_1/cmd_vel",
                 odom_topic="/originbot_1/odom"):
        super().__init__("explore_node")
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_timer(0.1, self._loop)

        self.duration = duration
        self.t0 = time.time()
        self.positions = []
        self.waypoints = []
        self.pos = None
        self.last_vx = 0.0
        self.stuck_count = 0
        self.last_wp_t = 0
        self.done = False

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pos = (p.x, p.y)
        self.positions.append((p.x, p.y, time.time() - self.t0))
        self.last_vx = msg.twist.twist.linear.x

        now = time.time()
        if now - self.last_wp_t > 1.5:
            self.waypoints.append((p.x, p.y))
            self.last_wp_t = now

    def _loop(self):
        if self.done:
            return
        if time.time() - self.t0 > self.duration:
            self.pub.publish(Twist())
            self.done = True
            self.get_logger().info(
                f"探索完成: {len(self.waypoints)} 航点, "
                f"{len(self.positions)} 位置, "
                f"{time.time()-self.t0:.1f}s"
            )
            return

        cmd = Twist()
        if abs(self.last_vx) < 0.02:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count > 10:
            cmd.angular.z = 0.6
            cmd.linear.x = 0.0
            if self.stuck_count > 25:
                self.stuck_count = 0
        else:
            cmd.linear.x = 0.25
            cmd.angular.z = 0.0

        self.pub.publish(cmd)


class NavNode(Node):
    """Session 2: 沿记忆航点直线导航（含超时 + 卡住跳过）"""

    def __init__(self, waypoints, cmd_topic="/originbot_1/cmd_vel",
                 odom_topic="/originbot_1/odom", timeout=60.0):
        super().__init__("nav_node")
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_timer(0.1, self._loop)

        self.wps = waypoints
        self.idx = 0
        self.pos = None
        self.yaw = 0.0
        self.t0 = time.time()
        self.timeout = timeout
        self.positions = []
        self.done = False
        self.wp_start_t = time.time()
        self.wp_best_dist = float("inf")
        self.skipped = 0

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pos = (p.x, p.y)
        self.positions.append((p.x, p.y, time.time() - self.t0))
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _finish(self, reason="done"):
        self.pub.publish(Twist())
        self.done = True
        t = time.time() - self.t0
        self.get_logger().info(
            f"导航{reason}: {t:.1f}s, {self.idx}/{len(self.wps)} 航点, "
            f"跳过 {self.skipped}")

    def _loop(self):
        if self.done or self.pos is None:
            return
        # 全局超时
        if time.time() - self.t0 > self.timeout:
            self._finish("超时")
            return
        if self.idx >= len(self.wps):
            self._finish("完成")
            return

        tx, ty = self.wps[self.idx]
        cx, cy = self.pos
        dist = math.hypot(tx - cx, ty - cy)

        if dist < 0.3:
            self.idx += 1
            self.wp_start_t = time.time()
            self.wp_best_dist = float("inf")
            return

        # 单航点卡住检测：5 秒没靠近就跳过
        self.wp_best_dist = min(self.wp_best_dist, dist)
        if time.time() - self.wp_start_t > 5.0 and dist >= self.wp_best_dist - 0.05:
            self.get_logger().info(f"跳过航点 {self.idx}（卡住）")
            self.skipped += 1
            self.idx += 1
            self.wp_start_t = time.time()
            self.wp_best_dist = float("inf")
            return

        dy, dx = ty - cy, tx - cx
        target_yaw = math.atan2(dy, dx)
        err = target_yaw - self.yaw
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi

        cmd = Twist()
        if abs(err) > 0.3:
            cmd.angular.z = max(-1.0, min(1.0, 1.2 * err))
            cmd.linear.x = 0.05
        else:
            cmd.linear.x = min(0.3, dist)
            cmd.angular.z = 0.5 * err
        self.pub.publish(cmd)


def path_len(pts):
    """轨迹总距离"""
    return sum(
        math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        for i in range(1, len(pts))
    )


def main():
    parser = argparse.ArgumentParser(description="robotmem ROS 2 导航记忆 Demo")
    parser.add_argument("duration", nargs="?", type=float, default=25.0,
                        help="探索时长（秒），默认 25")
    parser.add_argument("--prefix", default="/originbot_1",
                        help="Topic 前缀，如 /turtlebot3 或 /originbot_1")
    args = parser.parse_args()

    cmd_topic = f"{args.prefix}/cmd_vel"
    odom_topic = f"{args.prefix}/odom"

    # 清理旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    rclpy.init()

    print("=" * 55)
    print("robotmem ROS 2 Demo — 记忆驱动导航")
    print(f"Topics: {cmd_topic}, {odom_topic}")
    print("=" * 55)

    # ── Session 1: 探索 ──
    print(f"\n--- Session 1: 探索 ({args.duration}s) ---")
    exp = ExploreNode(duration=args.duration, cmd_topic=cmd_topic,
                      odom_topic=odom_topic)
    while rclpy.ok() and not exp.done:
        rclpy.spin_once(exp, timeout_sec=0.1)

    wps = exp.waypoints
    e_pos = exp.positions
    e_time = e_pos[-1][2] if e_pos else 0
    e_dist = path_len(e_pos)
    print(f"  航点: {len(wps)}, 距离: {e_dist:.2f}m, 用时: {e_time:.1f}s")

    # 存入 robotmem
    print("  存入 robotmem ...")
    with RobotMemory(db_path=DB_PATH, collection=COLLECTION,
                     embed_backend="none") as mem:
        with mem.session(context={"task": "navigation", "phase": "explore"}) as sid:
            try:
                mem.learn(
                    f"导航路线 {len(wps)} 航点",
                    context={
                        "spatial": {
                            "waypoints": wps,
                            "start": wps[0],
                            "end": wps[-1],
                        },
                        "params": {"distance": e_dist, "duration": e_time},
                        "task": {"type": "navigation", "result": "success"},
                    },
                )
                print("  learn ✓")
            except Exception as e:
                print(f"  learn ERROR: {e}")
                raise

            try:
                mem.save_perception(
                    f"探索轨迹 {len(e_pos)} 点",
                    perception_type="proprioceptive",
                    data=json.dumps({"positions": e_pos[:200], "waypoints": wps}),
                )
                print("  save_perception ✓")
            except Exception as e:
                print(f"  save_perception ERROR: {e}")
                raise

    exp.destroy_node()

    print("\n  暂停 3s（模拟重启）...")
    time.sleep(3)

    # ── Session 2: 记忆导航 ──
    print("\n--- Session 2: 记忆导航 ---")
    recalled_wps = []
    with RobotMemory(db_path=DB_PATH, collection=COLLECTION,
                     embed_backend="none") as mem:
        with mem.session(context={"task": "navigation", "phase": "memory"}) as sid:
            results = mem.recall("导航路线 航点")
            print(f"  recall: {len(results)} 条记忆")
            for r in results:
                ctx = r.get("context")
                if isinstance(ctx, str):
                    ctx = json.loads(ctx)
                if isinstance(ctx, dict):
                    w = ctx.get("spatial", {}).get("waypoints", [])
                    if w:
                        recalled_wps = [tuple(p) for p in w]
                        print(f"  提取航点: {len(recalled_wps)}")
                        break

    if not recalled_wps:
        print("  ERROR: 未能提取航点!")
        rclpy.shutdown()
        return

    nav = NavNode(recalled_wps, cmd_topic=cmd_topic, odom_topic=odom_topic)
    while rclpy.ok() and not nav.done:
        rclpy.spin_once(nav, timeout_sec=0.1)

    n_pos = nav.positions
    n_time = n_pos[-1][2] if n_pos else 0
    n_dist = path_len(n_pos)
    nav.destroy_node()
    rclpy.shutdown()

    # ── 结果 ──
    print("\n" + "=" * 55)
    print("结果对比")
    print("=" * 55)
    print(f"  Session 1 (探索): {e_time:6.1f}s  {e_dist:6.2f}m")
    print(f"  Session 2 (记忆): {n_time:6.1f}s  {n_dist:6.2f}m")
    if e_time > 0:
        print(f"  时间节省:        {(1 - n_time / e_time) * 100:6.1f}%")
    if e_dist > 0:
        print(f"  距离节省:        {(1 - n_dist / e_dist) * 100:6.1f}%")
    print(f"\n  DB: {DB_DIR}")


if __name__ == "__main__":
    main()
