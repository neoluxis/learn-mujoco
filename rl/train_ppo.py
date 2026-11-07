
# rl/train_ppo.py

from pathlib import Path
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from .cart_pole_env import CartPoleEnv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PPO model on the Cart-Pole environment."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a pretrained PPO model (.zip file) to continue training.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default='ppo',
        help="Name of the file to save the trained model.",
    )

    return parser.parse_args()

def main(args):

    # --------------------------------------------------------
    # 路径
    # --------------------------------------------------------

    project_dir = Path(__file__).resolve().parents[1]

    model_dir = project_dir / "results" / "models"
    log_dir = project_dir / "results" / "logs"
    
    pretrained_model_path = Path(args.model) if args.model else None

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 创建环境
    # --------------------------------------------------------

    env = CartPoleEnv(
        render_mode=None,
        frame_skip=10,
        max_force=20.0,
        max_episode_steps=500,
        init_x_range=0.5,
        init_theta_range=15.0,
        init_x_dot_range=0.05,
        init_theta_dot_range=0.05,
    )

    # 检查 Gymnasium 接口
    check_env(env, skip_render_check=True)

    env = Monitor(env)

    # --------------------------------------------------------
    # 创建 PPO
    # --------------------------------------------------------

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.0,
        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,
        tensorboard_log=str(log_dir),

        device="auto",
    )
    
    if pretrained_model_path:
        print(f"Loading pretrained model from: {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=env)

    # --------------------------------------------------------
    # 开始训练
    # --------------------------------------------------------

    model.learn(
        total_timesteps=500_000,
        tb_log_name="PPO_CartPole",
        progress_bar=True,
    )

    # --------------------------------------------------------
    # 保存模型
    # --------------------------------------------------------

    model.save(
        str(model_dir / args.save)
    )

    env.close()

    print("Training finished.")
    print(f"Model saved to: {model_dir}")


if __name__ == "__main__":
    main(parse_args())
