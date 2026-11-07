
import numpy as np


class StateFeedbackController:
    """
    基于极点配置的积分状态反馈控制器。

    状态:
        X = [x_error, theta, x_dot, theta_dot, integral_x]

    控制律:
        u = -K @ X
    """

    def __init__(
        self,
        K=None,
        max_force=20.0,
        x_ref=0.0,
    ):
        if K is None:
            K = np.array([
                -13.724910,
                -87.303568,
                -13.570469,
                -24.860120,
                -5.429763,
            ])

        self.K = np.asarray(K, dtype=float)
        self.max_force = float(max_force)
        self.x_ref = float(x_ref)

        self.integral_x = 0.0

    def reset(self):
        """重置控制器内部状态。"""
        self.integral_x = 0.0

    def update(
        self,
        x,
        theta,
        x_dot,
        theta_dot,
        dt,
        x_ref=None,
    ):
        """
        输入系统状态，返回控制力。

        Returns:
            force: 控制力 (N)
        """

        if x_ref is None:
            x_ref = self.x_ref

        # 位置误差
        error_x = x - x_ref

        # 积分状态
        self.integral_x += error_x * dt

        # 扩展状态向量
        X = np.array([
            error_x,
            theta,
            x_dot,
            theta_dot,
            self.integral_x,
        ])

        # 状态反馈
        force = -self.K @ X

        # 执行器限幅
        force = np.clip(
            force,
            -self.max_force,
            self.max_force,
        )

        return float(force)
