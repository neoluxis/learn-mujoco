
import mujoco
import mujoco.viewer

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，仅保存图片
import matplotlib.pyplot as plt

from pathlib import Path
import time


# =========================================================
# 1. 仿真参数
# =========================================================

MODEL_PATH = "cart_pole/cart_pole.xml"

SIM_TIME = 10.0

# PID 参数
KP = 50.0
KI = 0.0
KD = 2

MAX_FORCE = 20.0


# =========================================================
# 2. 创建实验目录
# =========================================================

runs_dir = Path("runs")
runs_dir.mkdir(exist_ok=True)

existing = [
    int(p.name[3:])
    for p in runs_dir.glob("pid*")
    if p.is_dir() and p.name[3:].isdigit()
]

run_id = max(existing, default=0) + 1

save_dir = runs_dir / f"pid{run_id:02d}"
save_dir.mkdir(parents=True)

print(f"Save directory: {save_dir}")


# =========================================================
# 3. 加载 MuJoCo 模型
# =========================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

dt = model.opt.timestep

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)


# =========================================================
# 4. 初始化系统
# =========================================================

data.qpos[0] = 0.0
data.qpos[1] = np.deg2rad(5.0)

data.qvel[:] = 0.0

mujoco.mj_forward(model, data)


# =========================================================
# 5. PID 控制器
# =========================================================

integral = 0.0


def pid_controller(theta, theta_dot):

    global integral

    # 目标角度为 0
    error = theta

    # 积分项
    integral += error * dt

    # 微分项
    derivative = theta_dot

    # PID 输出
    force = (
        KP * error
        + KI * integral
        + KD * derivative
    )

    # 输出限幅
    force = np.clip(
        force,
        -MAX_FORCE,
        MAX_FORCE
    )

    return force


# =========================================================
# 6. 数据记录
# =========================================================

history = {
    "time": [],
    "x": [],
    "theta": [],
    "x_dot": [],
    "theta_dot": [],
    "force": [],
}


def record_data(t, force):

    history["time"].append(t)

    history["x"].append(data.qpos[0])
    history["theta"].append(data.qpos[1])

    history["x_dot"].append(data.qvel[0])
    history["theta_dot"].append(data.qvel[1])

    history["force"].append(force)


# =========================================================
# 7. MuJoCo 仿真
# =========================================================

steps = int(SIM_TIME / dt)

with mujoco.viewer.launch_passive(model, data) as viewer:

    for i in range(steps):

        start = time.perf_counter()

        t = i * dt

        # 获取系统状态
        theta = data.qpos[1]
        theta_dot = data.qvel[1]

        # PID 控制
        force = pid_controller(
            theta,
            theta_dot
        )

        # 输出控制力
        data.ctrl[0] = force

        # 记录数据
        record_data(t, force)

        # 推进物理仿真
        mujoco.mj_step(model, data)

        # 更新 MuJoCo Viewer
        viewer.sync()

        # 控制实时仿真速度
        elapsed = time.perf_counter() - start

        if elapsed < dt:
            time.sleep(dt - elapsed)


print("Simulation completed.")


# =========================================================
# 8. 转换数据
# =========================================================

for key in history:
    history[key] = np.asarray(history[key])

t = history["time"]


# =========================================================
# 9. Matplotlib 绘图
# =========================================================

fig, axes = plt.subplots(
    4, 1,
    figsize=(11, 12),
    sharex=True
)

# ---------------------------------------------------------
# 9.1 摆杆角度
# ---------------------------------------------------------

axes[0].plot(
    t,
    np.rad2deg(history["theta"]),
    label="Theta"
)

axes[0].axhline(
    0,
    linestyle="--",
    label="Target"
)

axes[0].set_ylabel("Angle (deg)")
axes[0].set_title("Pole Angle")
axes[0].legend()
axes[0].grid(True)


# ---------------------------------------------------------
# 9.2 摆杆角速度
# ---------------------------------------------------------

axes[1].plot(
    t,
    np.rad2deg(history["theta_dot"]),
    label="Theta Dot"
)

axes[1].set_ylabel("Angular Velocity (deg/s)")
axes[1].set_title("Pole Angular Velocity")
axes[1].legend()
axes[1].grid(True)


# ---------------------------------------------------------
# 9.3 滑块位置
# ---------------------------------------------------------

axes[2].plot(
    t,
    history["x"],
    label="Cart Position"
)

axes[2].axhline(
    0,
    linestyle="--",
    label="Target"
)

axes[2].set_ylabel("Position (m)")
axes[2].set_title("Cart Position")
axes[2].legend()
axes[2].grid(True)


# ---------------------------------------------------------
# 9.4 控制力
# ---------------------------------------------------------

axes[3].plot(
    t,
    history["force"],
    label="Control Force"
)

axes[3].axhline(
    MAX_FORCE,
    linestyle="--",
    alpha=0.6
)

axes[3].axhline(
    -MAX_FORCE,
    linestyle="--",
    alpha=0.6
)

axes[3].set_ylabel("Force (N)")
axes[3].set_xlabel("Time (s)")
axes[3].set_title("PID Control Output")
axes[3].legend()
axes[3].grid(True)


# ---------------------------------------------------------
# 9.5 保存图片
# ---------------------------------------------------------

fig.suptitle(
    f"Cart-Pole PID | Kp={KP}, Ki={KI}, Kd={KD}",
    fontsize=14
)

fig.tight_layout()

fig.savefig(
    save_dir / "pid_result.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# 10. 保存原始数据
# =========================================================

np.savez(
    save_dir / "pid_data.npz",
    **history
)


# =========================================================
# 11. 实验结果
# =========================================================

print("\nExperiment completed.")
print(f"Image: {save_dir / 'pid_result.png'}")
print(f"Data:  {save_dir / 'pid_data.npz'}")

theta = history["theta"]
x = history["x"]
force = history["force"]

# 最大摆杆偏转角
max_angle = np.rad2deg(np.max(np.abs(theta)))

# 最终角度误差
final_error = np.rad2deg(np.abs(theta[-1]))

# 最大滑块位移
max_position = np.max(np.abs(x))

# 控制力饱和比例
saturation_ratio = np.mean(
    np.abs(force) >= MAX_FORCE * 0.999
) * 100

print(f"最大摆角: {max_angle:.4f} deg")
print(f"最终角度误差: {final_error:.4f} deg")
print(f"最大滑块位移: {max_position:.4f} m")
print(f"控制力饱和比例: {saturation_ratio:.2f}%")