import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("cart_pole/cart_pole.xml")
data = mujoco.MjData(model)

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)

# 初始化：摆杆偏离竖直方向 5°
data.qpos[0] = 0.0
data.qpos[1] = np.deg2rad(5)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # 暂时施加零输入，观察系统自然运动
        data.ctrl[0] = 0.0

        mujoco.mj_step(model, data)

        viewer.sync()
        
        time.sleep(0.01)