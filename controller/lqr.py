
import numpy as np
from scipy.linalg import solve_continuous_are


class LQRController:
    """
    基于连续时间 CARE 的积分 LQR 控制器。

    状态:
        X = [
            x - x_ref,
            theta,
            x_dot,
            theta_dot,
            integral_x
        ]

    控制律:
        u = -K @ X
    """

    def __init__(
        self,
        Q=None,
        R=None,
        max_force=20.0,
        x_ref=0.0,
    ):

        # 连续时间线性化模型
        self.A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -5.683755, -0.046760, 0.049281],
            [0.0, 16.329248, 0.049281, -0.141583],
        ])

        self.B = np.array([
            [0.0],
            [0.0],
            [0.935190],
            [-0.985620],
        ])

        # 默认权重矩阵
        if Q is None:
            Q = np.diag([
                10.0,   # 位置误差
                100.0,  # 摆杆角度
                1.0,    # 滑块速度
                1.0,    # 摆杆角速度
                50.0,   # 位置误差积分
            ])

        if R is None:
            R = np.array([[1.0]])

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        # 构建增广系统
        self.A_aug = np.zeros((5, 5))
        self.A_aug[:4, :4] = self.A
        self.A_aug[4, 0] = 1.0

        self.B_aug = np.zeros((5, 1))
        self.B_aug[:4, :] = self.B

        # 求解连续时间代数 Riccati 方程
        P = solve_continuous_are(
            self.A_aug,
            self.B_aug,
            self.Q,
            self.R,
        )

        # 最优反馈增益
        self.K = np.linalg.solve(
            self.R,
            self.B_aug.T @ P,
        )

        self.max_force = float(max_force)
        self.x_ref = float(x_ref)

        self.reset()

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
        """根据当前系统状态计算控制力。"""

        if x_ref is None:
            x_ref = self.x_ref

        # 位置误差
        error_x = x - x_ref

        # 积分状态
        self.integral_x += error_x * dt

        # 增广状态
        X = np.array([
            error_x,
            theta,
            x_dot,
            theta_dot,
            self.integral_x,
        ])

        # LQR 控制律
        force = (-self.K @ X).item()

        # 执行器限幅
        force = np.clip(
            force,
            -self.max_force,
            self.max_force,
        )

        return force

    def get_gain(self):
        """返回 LQR 反馈增益。"""
        return self.K.copy()
