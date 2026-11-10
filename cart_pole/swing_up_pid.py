"""当前 cart_pole.xml 的能量起摆与位置/角度串级 PID 稳摆。

运行：python cart_pole/swing_up_pid.py [--headless]
原始关节角 0° 为下垂，180° 为直立；控制器 theta=0 为直立。
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "cart_pole/cart_pole.xml"


@dataclass
class Config:
    duration: float = 20.0
    x_ref: float = 0.0
    max_force: float = 20.0
    # 起摆：能量差 -> 目标小车加速度 -> 动力学换算成推力。
    energy_gain: float = 0.55
    swing_position_gain: float = 2.5
    swing_velocity_gain: float = 1.25
    max_acceleration: float = 12.0
    kick_acceleration: float = 3.0
    kick_duration: float = 0.2
    # 稳摆：位置 PID 输出目标倾角；角度 PID 输出推力。
    position_pid: tuple = (0.16, 0.06, 0.155)
    angle_pid: tuple = (90.0, 0.5, 25.0)
    max_target_angle: float = np.deg2rad(12.0)
    capture_angle: float = np.deg2rad(15.0)
    capture_speed: float = 1.5
    release_angle: float = np.deg2rad(30.0)
    rail_margin: float = 0.02
    # 连续满足以下条件 2 秒，才判为稳定。
    stable_angle: float = np.deg2rad(1.0)
    stable_position: float = 0.02
    stable_cart_speed: float = 0.03
    stable_pole_speed: float = 0.05
    stable_duration: float = 2.0


def upright_error(q):
    """相对直立的误差角，限制在 [-pi, pi]，不对它做差分求速度。"""
    return float(np.arctan2(np.sin(q - np.pi), np.cos(q - np.pi)))


class LimitedPID:
    """使用测量速度做微分，并在输出饱和时条件积分。"""

    def __init__(self, gains, limit):
        self.kp, self.ki, self.kd = gains
        self.limit = limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.saturated = False

    def update(self, error, rate, dt, integrate=True):
        candidate = self.integral
        if integrate and self.ki > 0:
            candidate += error * dt
            candidate = float(np.clip(candidate, -self.limit / self.ki,
                                      self.limit / self.ki))
        raw = self.kp * error + self.ki * candidate + self.kd * rate
        # 允许反向积分解除饱和，禁止同向积分进一步堆积。
        if abs(raw) > self.limit and raw * error > 0:
            candidate = self.integral
            raw = self.kp * error + self.ki * candidate + self.kd * rate
        self.integral = candidate
        self.saturated = abs(raw) >= self.limit
        return float(np.clip(raw, -self.limit, self.limit))


class SwingUpCascadePID:
    def __init__(self, model, config):
        self.config = config
        if (model.nq, model.nv, model.nu) != (2, 2, 1):
            raise ValueError("This example requires the two-joint cart-pole model.")
        if not np.isclose(model.actuator_gear[0, 0], 1.0):
            raise ValueError("The cart motor must have gear=1 (control in N).")
        ctrl_min, ctrl_max = model.actuator_ctrlrange[0]
        self.max_force = min(config.max_force, float(ctrl_max), -float(ctrl_min))
        if self.max_force <= 0:
            raise ValueError("The cart motor must support positive and negative force.")

        # 在下垂位置读取质量矩阵，包含球体质量和关节 armature，
        # 避免将摆杆 geom 的 mass=0.2 误认为整个摆体质量。
        probe = mujoco.MjData(model)
        mujoco.mj_forward(model, probe)
        mass = np.empty((2, 2))
        mujoco.mj_fullM(model, probe, mass)
        self.total_mass = mass[0, 0]
        self.first_moment = -mass[0, 1]  # m*l
        self.inertia = mass[1, 1]       # 绕铰点的惯量
        self.gravity = -model.opt.gravity[2]
        self.cart_damping, self.pole_damping = model.dof_damping
        self.target_energy = self.first_moment * self.gravity

        self.position_pid = LimitedPID(config.position_pid, config.max_target_angle)
        self.angle_pid = LimitedPID(config.angle_pid, self.max_force)
        self.mode = "swing_up"
        self.theta_ref = np.nan

    def energy(self, theta, theta_dot):
        return (0.5 * self.inertia * theta_dot**2
                + self.target_energy * np.cos(theta))

    def update(self, x, theta, x_dot, theta_dot, dt, elapsed):
        cfg = self.config
        if self.mode == "balance" and abs(theta) > cfg.release_angle:
            self.mode = "swing_up"
            self.position_pid.reset()
            self.angle_pid.reset()
        if (self.mode == "swing_up" and abs(theta) < cfg.capture_angle
                and abs(theta_dot) < cfg.capture_speed):
            self.mode = "balance"
            self.position_pid.reset()
            self.angle_pid.reset()

        if self.mode == "balance":
            # 小车偏右时给负倾角，使其先向右接摆，再向左回中。
            self.theta_ref = -self.position_pid.update(
                x - cfg.x_ref, x_dot, dt,
                integrate=not self.angle_pid.saturated,
            )
            # 当前模型推力对直立角加速度的作用为负，故使用
            # theta-theta_ref；D 项直接使用角速度，避免参考值跳变冲击。
            return self.angle_pid.update(
                theta - self.theta_ref, theta_dot, dt,
            )

        self.theta_ref = np.nan
        c, s = np.cos(theta), np.sin(theta)
        h, j = self.first_moment, self.inertia
        acceleration = (
            cfg.energy_gain * (self.energy(theta, theta_dot) - self.target_energy)
            * theta_dot * c
            - cfg.swing_position_gain * (x - cfg.x_ref)
            - cfg.swing_velocity_gain * x_dot
        )
        # 完全下垂且静止是能量控制的零输出点，需要短暂启动激励。
        if elapsed < cfg.kick_duration:
            acceleration = cfg.kick_acceleration
        acceleration = float(np.clip(acceleration, -cfg.max_acceleration,
                                     cfg.max_acceleration))
        force = (
            (self.total_mass - h*h*c*c/j) * acceleration
            + h*h*self.gravity*c*s/j
            - h*self.pole_damping*c*theta_dot/j
            - h*s*theta_dot**2
            + self.cart_damping*x_dot
        )
        return float(np.clip(force, -self.max_force, self.max_force))


def simulate(config, *, render=False, initial_angle=0.0, initial_x=0.0):
    """initial_angle 为原始关节角，单位 rad；返回轨迹和验收指标。"""
    if not np.isfinite(config.duration) or config.duration <= 0:
        raise ValueError("duration must be finite and positive")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    controller = SwingUpCascadePID(model, config)
    rail_low, rail_high = model.jnt_range[0]
    safe_low = rail_low + config.rail_margin
    safe_high = rail_high - config.rail_margin
    if not safe_low < initial_x < safe_high:
        raise ValueError("Initial cart position is outside the rail safety margin.")
    if not safe_low < config.x_ref < safe_high:
        raise ValueError("Cart reference is outside the rail safety margin.")
    data.qpos[:] = [initial_x, initial_angle]
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    history = {key: [] for key in (
        "time", "x", "theta", "x_dot", "theta_dot", "force", "theta_ref",
        "energy", "mode",
    )}
    switches = []
    stable_since = None
    first_capture = None
    reason = "completed"
    max_position = abs(initial_x)
    sync_interval = max(1, round(1.0 / (60 * dt)))

    with ExitStack() as stack:
        viewer = None
        if render:
            from mujoco import viewer as mujoco_viewer

            viewer = stack.enter_context(mujoco_viewer.launch_passive(model, data))
            viewer.cam.lookat[:] = [0, 0, 1.2]
            viewer.cam.distance = 4.5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -15

        for step in range(int(np.ceil(config.duration / dt))):
            start = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                reason = "viewer_closed"
                break
            x, q = data.qpos
            x_dot, theta_dot = data.qvel
            theta = upright_error(q)
            old_mode = controller.mode
            force = controller.update(x, theta, x_dot, theta_dot, dt, data.time)
            if controller.mode != old_mode:
                switches.append({"time": float(data.time), "mode": controller.mode})
                if controller.mode == "balance" and first_capture is None:
                    first_capture = float(data.time)
            values = (data.time, x, theta, x_dot, theta_dot, force,
                      controller.theta_ref, controller.energy(theta, theta_dot),
                      int(controller.mode == "balance"))
            for key, value in zip(history, values):
                history[key].append(value)

            data.ctrl[0] = force
            mujoco.mj_step(model, data)
            max_position = max(max_position, abs(float(data.qpos[0])))
            if not np.all(np.isfinite(np.r_[data.qpos, data.qvel])):
                reason = "nonfinite_state"
                break
            # 不让一次撞限位后的反弹被当成成功起摆。
            if not safe_low < data.qpos[0] < safe_high:
                reason = "rail_margin_reached"
                break

            settled = (
                controller.mode == "balance"
                and abs(upright_error(data.qpos[1])) < config.stable_angle
                and abs(data.qpos[0] - config.x_ref) < config.stable_position
                and abs(data.qvel[0]) < config.stable_cart_speed
                and abs(data.qvel[1]) < config.stable_pole_speed
            )
            if settled:
                if stable_since is None:
                    stable_since = float(data.time)
            else:
                stable_since = None

            if viewer is not None:
                if step % sync_interval == 0:
                    viewer.sync()
                remaining = dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)

    history = {key: np.asarray(values) for key, values in history.items()}
    summary = {
        "stable": bool(reason == "completed" and stable_since is not None
                       and data.time - stable_since >= config.stable_duration),
        "termination": reason,
        "duration_s": float(data.time),
        "first_capture_s": first_capture,
        "stable_since_s": stable_since,
        "final_angle_deg": float(np.rad2deg(upright_error(data.qpos[1]))),
        "final_position_m": float(data.qpos[0]),
        "final_cart_speed_m_s": float(data.qvel[0]),
        "final_pole_speed_rad_s": float(data.qvel[1]),
        "max_abs_position_m": max_position,
        "max_abs_force_n": float(np.max(np.abs(history["force"]), initial=0.0)),
        "target_energy_j": float(controller.target_energy),
        "rail_limits_m": [float(rail_low), float(rail_high)],
        "mode_switches": switches,
        "initial_angle_deg": float(np.rad2deg(initial_angle)),
        "initial_x_m": initial_x,
        "config": asdict(config),
    }
    return history, summary


def save_results(history, summary, output_dir, *, plot=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "trajectory.npz", **history)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    if not plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = history["time"]
    display_angle = np.rad2deg(history["theta"])
    # +/-180° 处断开连线，避免把角度包络跳变画成物理突变。
    jumps = np.flatnonzero(np.abs(np.diff(history["theta"])) > np.pi) + 1
    display_angle[jumps] = np.nan
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)
    curves = (
        (display_angle, "Pole angle (wrapped)", "deg"),
        (history["x"], "Cart position", "m"),
        (history["theta_dot"], "Pole angular velocity", "rad/s"),
        (history["x_dot"], "Cart velocity", "m/s"),
        (history["force"], "Motor force", "N"),
        (history["energy"], "Pole energy", "J"),
        (np.rad2deg(history["theta_ref"]), "Outer PID target angle", "deg"),
        (history["mode"], "Controller mode", ""),
    )
    for ax, (values, title, unit) in zip(axes.flat, curves):
        ax.plot(t, values, linewidth=1)
        ax.set(title=title, ylabel=unit)
        ax.grid(True, alpha=0.3)
        for event in summary["mode_switches"]:
            ax.axvline(event["time"], color="tab:green", linestyle=":", alpha=0.5)
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.4)
    axes[0, 1].axhline(summary["config"]["x_ref"], color="black", linestyle="--")
    for bound in summary["rail_limits_m"]:
        axes[0, 1].axhline(bound, color="tab:red", linestyle="--", alpha=0.5)
    axes[2, 1].axhline(summary["target_energy_j"], color="black", linestyle="--")
    axes[3, 1].set_yticks([0, 1], ["Swing up", "Cascade PID"])
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Cart-Pole: Energy Swing-Up + Cascaded Position/Angle PID")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Disable the viewer.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--initial-angle", type=float, default=0.0,
                        help="Raw hinge angle in degrees: 0=down, 180=upright.")
    parser.add_argument("--initial-x", type=float, default=0.0)
    parser.add_argument("--x-ref", type=float, default=0.0)
    parser.add_argument("--position-pid", type=float, nargs=3,
                        default=(0.16, 0.06, 0.155), metavar=("KP", "KI", "KD"))
    parser.add_argument("--angle-pid", type=float, nargs=3,
                        default=(90.0, 0.5, 25.0), metavar=("KP", "KI", "KD"))
    parser.add_argument("--energy-gain", type=float, default=0.55)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results/swing_up_pid")
    parser.add_argument("--no-plot", action="store_true", help="Save data without plotting.")
    args = parser.parse_args()
    for name in ("duration", "initial_angle", "initial_x", "x_ref", "energy_gain"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.duration <= 0 or args.energy_gain < 0:
        parser.error("duration must be positive and energy-gain must be nonnegative")
    if any(not np.isfinite(gain) or gain < 0
           for gain in (*args.position_pid, *args.angle_pid)):
        parser.error("PID gains must be finite and nonnegative")
    config = Config(duration=args.duration, x_ref=args.x_ref,
                    position_pid=tuple(args.position_pid), angle_pid=tuple(args.angle_pid),
                    energy_gain=args.energy_gain)
    history, summary = simulate(
        config, render=not args.headless,
        initial_angle=np.deg2rad(args.initial_angle), initial_x=args.initial_x,
    )
    save_results(history, summary, args.output_dir, plot=not args.no_plot)
    for event in summary["mode_switches"]:
        print(f"t={event['time']:.3f}s -> {event['mode']}")
    print(f"Stable: {summary['stable']} ({summary['termination']})")
    print(f"Final angle: {summary['final_angle_deg']:.6f} deg")
    print(f"Final position: {summary['final_position_m']:.6f} m")
    print(f"Maximum cart displacement: {summary['max_abs_position_m']:.3f} m")
    print(f"Results: {args.output_dir.resolve()}")
    return 0 if summary["stable"] or summary["termination"] == "viewer_closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
