
# rl/cartpole_env.py

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.mujoco import MujocoEnv


# XML 路径
XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "cart_pole"
    / "cart_pole.xml"
)


class CartPoleEnv(gym.Env):
    """
    MuJoCo Cart-Pole Gymnasium Environment.

    Observation:
        [x_error, theta, x_dot, theta_dot]

    Action:
        normalized force in [-1, 1]

    Reward:
        position + angle + velocity + action penalty
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode=None,
        frame_skip=10,
        max_force=20.0,
        max_episode_steps=500,
        x_ref=0.0,
        max_cart_position=1.5,
        max_pole_angle=np.deg2rad(85),
        init_theta_range=np.deg2rad(3),
        init_x_range=0.05,
        init_x_dot_range=0.05,
        init_theta_dot_range=0.05,
    ):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode

        self.frame_skip = int(frame_skip)
        self.max_force = float(max_force)
        self.max_episode_steps = int(max_episode_steps)

        self.x_ref = float(x_ref)
        self.max_cart_position = float(max_cart_position)
        self.max_pole_angle = float(max_pole_angle)

        # ----------------------------------------------------
        # MuJoCo 环境
        # ----------------------------------------------------

        self.sim = MujocoEnv(
            XML_PATH,
            render=(render_mode == "human"),
        )

        self.dt = self.sim.dt
        self.control_dt = self.dt * self.frame_skip

        # ----------------------------------------------------
        # Action Space
        # ----------------------------------------------------

        # a in [-1, 1]
        # force = a * max_force

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Observation Space
        # ----------------------------------------------------

        # 归一化尺度:
        # x_error / 0.5
        # theta / 0.2
        # x_dot / 2.0
        # theta_dot / 4.0

        self.obs_scale = np.array(
            [0.5, 0.2, 2.0, 4.0],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Reward Parameters
        # ----------------------------------------------------

        self.w_x = 1.5
        self.w_theta = 0.5
        self.w_x_dot = 0.1
        self.w_theta_dot = 0.1
        self.w_action = 0.001

        self.alive_reward = 1.0
        self.termination_penalty = 10.0

        # ----------------------------------------------------
        # Initial State Randomization
        # ----------------------------------------------------

        self.init_x_range = (-init_x_range, init_x_range)
        self.init_theta_range = (
            -init_theta_range,
            init_theta_range,
        )

        self.init_x_dot_range = (-init_x_dot_range, init_x_dot_range)
        self.init_theta_dot_range = (-init_theta_dot_range, init_theta_dot_range)

        self.elapsed_steps = 0

    # ========================================================
    # Observation
    # ========================================================

    def _get_state(self):
        state = self.sim.get_state()

        x = float(state["qpos"][0])
        theta = float(state["qpos"][1])

        x_dot = float(state["qvel"][0])
        theta_dot = float(state["qvel"][1])

        return x, theta, x_dot, theta_dot

    def _get_obs(self):
        x, theta, x_dot, theta_dot = self._get_state()

        raw_obs = np.array(
            [
                x - self.x_ref,
                theta,
                x_dot,
                theta_dot,
            ],
            dtype=np.float32,
        )

        obs = raw_obs / self.obs_scale

        return obs.astype(np.float32)

    # ========================================================
    # Reward
    # ========================================================

    def _get_reward(self, action):
        x, theta, x_dot, theta_dot = self._get_state()

        x_error = x - self.x_ref

        # 使用归一化后的状态计算代价
        cost = (
            self.w_x * (x_error / 0.5) ** 2
            + self.w_theta * (theta / 0.2) ** 2
            + self.w_x_dot * (x_dot / 2.0) ** 2
            + self.w_theta_dot * (theta_dot / 4.0) ** 2
            + self.w_action * float(action[0]) ** 2
        )

        return float(self.alive_reward - cost)

    # ========================================================
    # Termination
    # ========================================================

    def _is_terminated(self):
        x, theta, _, _ = self._get_state()

        # 摆杆倾倒
        pole_fallen = abs(theta) > self.max_pole_angle

        # 滑块超出允许范围
        cart_out_of_range = (
            abs(x - self.x_ref) > self.max_cart_position
        )

        return pole_fallen or cart_out_of_range

    # ========================================================
    # Reset
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.elapsed_steps = 0

        # 重置 MuJoCo
        self.sim.reset()

        # 随机初始状态
        x = self.np_random.uniform(*self.init_x_range)
        theta = self.np_random.uniform(*self.init_theta_range)

        x_dot = self.np_random.uniform(*self.init_x_dot_range)
        theta_dot = self.np_random.uniform(
            *self.init_theta_dot_range
        )

        # 允许外部指定初始状态
        if options is not None:
            x = options.get("x", x)
            theta = options.get("theta", theta)
            x_dot = options.get("x_dot", x_dot)
            theta_dot = options.get("theta_dot", theta_dot)

        self.sim.set_state(
            qpos=[x, theta],
            qvel=[x_dot, theta_dot],
        )

        obs = self._get_obs()

        info = {
            "x": x,
            "theta": theta,
            "x_dot": x_dot,
            "theta_dot": theta_dot,
        }

        return obs, info

    # ========================================================
    # Step
    # ========================================================

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(self.action_space.shape)

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # 归一化动作映射到物理控制力
        force = float(action[0]) * self.max_force

        # 一个 RL step 对应多个 MuJoCo physics steps
        self.sim.set_control([force])

        terminated = False

        for _ in range(self.frame_skip):

            self.sim.step()

            # 每个物理步检查是否已经失败
            if self._is_terminated():
                terminated = True
                break

        self.elapsed_steps += 1

        truncated = (
            self.elapsed_steps >= self.max_episode_steps
        )

        obs = self._get_obs()

        reward = self._get_reward(action)

        # 失败惩罚
        if terminated:
            reward -= self.termination_penalty

        x, theta, x_dot, theta_dot = self._get_state()

        info = {
            "x": x,
            "theta": theta,
            "x_dot": x_dot,
            "theta_dot": theta_dot,
            "force": force,
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # Render
    # ========================================================

    def render(self):

        if self.render_mode == "human":
            return self.sim.render()

        return None

    # ========================================================
    # Close
    # ========================================================

    def close(self):

        if hasattr(self, "sim") and self.sim is not None:
            self.sim.close()
            self.sim = None


def main():
    import numpy as np

    from gymnasium.utils.env_checker import check_env
    env = CartPoleEnv(
        render_mode=None,
        frame_skip=10,
    )

    # Gymnasium 接口检查
    check_env(env, skip_render_check=True)

    obs, info = env.reset(seed=42)

    print("Initial observation:", obs)
    print("Initial info:", info)

    for i in range(1000):

        # 随机动作
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if i % 50 == 0:
            print(
                f"Step: {i:4d} | "
                f"Reward: {reward:8.4f} | "
                f"X: {info['x']:8.4f} | "
                f"Theta: {np.rad2deg(info['theta']):8.3f} deg"
            )

        if terminated or truncated:
            print(
                f"Episode ended at step {i}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )

            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
