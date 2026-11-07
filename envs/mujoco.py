
import mujoco
import mujoco.viewer
import numpy as np


class MujocoEnv:
    """
    MuJoCo 仿真环境封装。

    负责：
        1. 加载模型
        2. 初始化仿真数据
        3. 设置控制输入
        4. 执行仿真
        5. 获取状态
        6. 管理 Viewer
    """

    def __init__(self, xml_path, render=True):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep

        self.viewer = None
        self.render_enabled = render

        if self.render_enabled:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

    def reset(self):
        """重置仿真状态。"""

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_state(self, qpos=None, qvel=None):
        """设置广义位置和速度。"""

        if qpos is not None:
            self.data.qpos[:] = qpos

        if qvel is not None:
            self.data.qvel[:] = qvel

        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        """
        获取仿真状态。

        Returns
        -------
        dict
            time, qpos, qvel, qacc, ctrl
        """

        return {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "qacc": self.data.qacc.copy(),
            "ctrl": self.data.ctrl.copy(),
        }

    def set_control(self, control):
        """设置执行器控制输入。"""

        control = np.asarray(control, dtype=float)

        if control.size != self.model.nu:
            raise ValueError(
                f"控制输入维度错误：需要 {self.model.nu}，"
                f"实际为 {control.size}"
            )

        self.data.ctrl[:] = control

    def step(self, n=1):
        """执行 n 步仿真。"""

        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        """同步 Viewer。"""

        if self.viewer is not None:
            if not self.viewer.is_running():
                return False

            self.viewer.sync()
            return True

        return False

    def close(self):
        """关闭 Viewer。"""

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def time(self):
        return self.data.time

    @property
    def nq(self):
        return self.model.nq

    @property
    def nv(self):
        return self.model.nv

    @property
    def nu(self):
        return self.model.nu
