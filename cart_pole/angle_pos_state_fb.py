
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from envs.mujoco import MujocoEnv
from utils.plots import plot_curves
from controller.state_fb import StateFeedbackController


XML_PATH = Path("cart_pole/cart_pole.xml")

SIM_TIME = 10.0

MAX_FORCE = 20.0

# 初始状态
INIT_X = 0
INIT_THETA = np.deg2rad(5)

# 位置参考值
X_REF = 0

# 是否实时显示 MuJoCo Viewer
RENDER = True


def main():

    # 数据记录
    history = {
        "time": [],
        "x": [],
        "theta": [],
        "x_dot": [],
        "theta_dot": [],
        "force": [],
    }

    # 创建控制器
    controller = StateFeedbackController(
        max_force=MAX_FORCE,
        x_ref=X_REF,
    )

    controller.reset()

    # 创建 MuJoCo 环境
    with MujocoEnv(XML_PATH, render=RENDER) as env:

        env.reset()

        env.set_state(
            qpos=[INIT_X, INIT_THETA],
            qvel=[0.0, 0.0],
        )

        dt = env.dt
        steps = int(SIM_TIME / dt)

        # 仿真循环
        for _ in range(steps):

            loop_start = time.perf_counter()

            state = env.get_state()

            x = state["qpos"][0]
            theta = state["qpos"][1]

            x_dot = state["qvel"][0]
            theta_dot = state["qvel"][1]

            # 控制器计算
            force = controller.update(
                x=x,
                theta=theta,
                x_dot=x_dot,
                theta_dot=theta_dot,
                dt=dt,
            )

            # 设置执行器输入
            env.set_control([force])

            # 记录数据
            history["time"].append(state["time"])
            history["x"].append(x)
            history["theta"].append(theta)
            history["x_dot"].append(x_dot)
            history["theta_dot"].append(theta_dot)
            history["force"].append(force)

            # 执行物理仿真
            env.step()

            # 更新 Viewer
            if RENDER:

                if not env.render():
                    break

                # 尽量按照仿真时间实时运行
                elapsed = time.perf_counter() - loop_start
                remaining = dt - elapsed

                if remaining > 0:
                    time.sleep(remaining)

    # 数据处理
    for key in history:
        history[key] = np.asarray(history[key])

    # 绘图
    time_data = history["time"]

    plot_curves(
        time=time_data,

        data_list=[
            np.rad2deg(history["theta"]),
            history["x"],
            history["force"],
            np.rad2deg(history["theta_dot"]),
            history["x_dot"],
        ],

        titles=[
            "Pole Angle",
            "Cart Position",
            "Control Force",
            "Pole Angular Velocity",
            "Cart Velocity",
        ],

        total_title="Cart-Pole Integral State Feedback",

        rows=3,
        cols=2,

        ylabels=[
            "Angle (deg)",
            "Position (m)",
            "Force (N)",
            "Angular Velocity (deg/s)",
            "Velocity (m/s)",
        ],

        save_path="results/state_feedback.png",
    )

    plt.close("all")

    # 保存原始数据
    Path("results").mkdir(exist_ok=True)

    np.savez(
        "results/state_feedback.npz",
        **history,
    )

    print("Simulation finished.")
    print("Results saved to results/")


if __name__ == "__main__":
    main()
