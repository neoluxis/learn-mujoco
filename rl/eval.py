
# rl/evaluate.py

import time
from pathlib import Path
import argparse

import numpy as np
from stable_baselines3 import PPO

from .cart_pole_env import CartPoleEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model on the Cart-Pole environment."
    )

    parser.add_argument(
        "model",
        type=str,
        default=None,
        help="Path to the trained PPO model (.zip file).",
    )

    return parser.parse_args()

def main(args):

    project_dir = Path(__file__).resolve().parents[1]

    model_path = Path(args.model)

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model = PPO.load(model_path)

    # --------------------------------------------------------
    # 创建可视化环境
    # --------------------------------------------------------

    env = CartPoleEnv(
        render_mode="human",
        frame_skip=10,
        max_force=20.0,
        max_episode_steps=500,
    )

    # 固定初始状态，方便重复评估
    obs, info = env.reset(
        seed=42,
        options={
            "x": 0.5,
            "theta": np.deg2rad(-15),
            "x_dot": 0.0,
            "theta_dot": 0.0,
        },
    )

    episode_reward = 0.0

    try:

        while True:

            loop_start = time.perf_counter()

            # 确定性策略
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_reward += reward

            # 更新 Viewer
            render_result = env.render()

            if render_result is False:
                break

            print(
                f"x={info['x']:+.3f} m | "
                f"theta={np.rad2deg(info['theta']):+.2f} deg | "
                f"force={info['force']:+.2f} N | "
                f"reward={reward:+.3f}"
            )

            # 按控制周期运行
            remaining = (
                env.control_dt
                - (time.perf_counter() - loop_start)
            )

            if remaining > 0:
                time.sleep(remaining)

            if terminated or truncated:

                print(
                    f"\nEpisode finished: "
                    f"terminated={terminated}, "
                    f"truncated={truncated}"
                )

                print(f"Episode reward: {episode_reward:.3f}")

                obs, info = env.reset()
                episode_reward = 0.0

    except KeyboardInterrupt:
        print("\nEvaluation stopped.")

    finally:
        env.close()


if __name__ == "__main__":
    main(parse_args())
