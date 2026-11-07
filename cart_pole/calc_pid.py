
import mujoco
import numpy as np

from scipy.signal import place_poles


# ============================================================
# 配置
# ============================================================

XML_PATH = "cart_pole/cart_pole.xml"

# 有限差分步长
EPS = 1e-5

# 状态定义：
# [x, theta, x_dot, theta_dot]

# 期望闭环极点
# 先采用相对保守的极点配置
POLES = [-1.0, -2.0, -2.5, -3.0, -3.5]


# ============================================================
# 加载模型
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

assert model.nq == 2
assert model.nv == 2
assert model.nu == 1

print("========== Model Parameters ==========")
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("timestep:", model.opt.timestep)
print("gravity:", model.opt.gravity)


# ============================================================
# 定义非线性状态方程
# ============================================================

def dynamics(state, control):
    """
    输入：
        state = [x, theta, x_dot, theta_dot]
        control = [force]

    输出：
        state_dot = [x_dot, theta_dot, x_ddot, theta_ddot]
    """

    mujoco.mj_resetData(model, data)

    data.qpos[:] = state[:2]
    data.qvel[:] = state[2:]
    data.ctrl[:] = control

    # 计算当前状态下的动力学
    mujoco.mj_forward(model, data)

    return np.concatenate([
        data.qvel.copy(),
        data.qacc.copy(),
    ])


# ============================================================
# 数值计算 A、B
# ============================================================

def linearize():

    x0 = np.zeros(4)
    u0 = np.zeros(1)

    f0 = dynamics(x0, u0)

    A = np.zeros((4, 4))
    B = np.zeros((4, 1))

    # --------------------------------------------------------
    # A = df / dx
    # --------------------------------------------------------

    for i in range(4):

        dx = np.zeros(4)
        dx[i] = EPS

        f_plus = dynamics(x0 + dx, u0)
        f_minus = dynamics(x0 - dx, u0)

        A[:, i] = (
            f_plus - f_minus
        ) / (2 * EPS)

    # --------------------------------------------------------
    # B = df / du
    # --------------------------------------------------------

    du = np.array([EPS])

    f_plus = dynamics(x0, u0 + du)
    f_minus = dynamics(x0, u0 - du)

    B[:, 0] = (
        f_plus - f_minus
    ) / (2 * EPS)

    return A, B


A, B = linearize()


# ============================================================
# 设计带位置积分的状态反馈控制器
# ============================================================

# 原始系统：
# x_dot = A*x + B*u
#
# 增加位置积分状态：
# xi_dot = x
#
# 扩展状态：
# [x, theta, x_dot, theta_dot, xi]

A_aug = np.zeros((5, 5))
B_aug = np.zeros((5, 1))

A_aug[:4, :4] = A
B_aug[:4, :] = B

# xi_dot = x
A_aug[4, 0] = 1.0

# 极点配置
result = place_poles(
    A_aug,
    B_aug,
    POLES,
)

K = result.gain_matrix[0]


# ============================================================
# 输出结果
# ============================================================

A_cl = A_aug - B_aug @ K.reshape(1, -1)

print("\n========== Linearized Model ==========")

np.set_printoptions(
    precision=6,
    suppress=True,
)

print("\nA =")
print(A)

print("\nB =")
print(B)

print("\n========== State Feedback ==========")

print("K =", K)

print("\nControl law:")
print(
    "u = "
    f"{-K[0]:.6f} * x "
    f"{-K[1]:+.6f} * theta "
    f"{-K[2]:+.6f} * x_dot "
    f"{-K[3]:+.6f} * theta_dot "
    f"{-K[4]:+.6f} * integral_x"
)

print("\n========== Closed-loop Poles ==========")

eigenvalues = np.linalg.eigvals(A_cl)

for i, eig in enumerate(eigenvalues):
    print(f"Pole {i + 1}: {eig}")

print("\nStable:", np.all(np.real(eigenvalues) < 0))

print("\n========== Controllability ==========")

controllability = np.hstack([
    np.linalg.matrix_power(A_aug, i) @ B_aug
    for i in range(5)
])

print(
    "Rank:",
    np.linalg.matrix_rank(controllability),
    "/ 5"
)
