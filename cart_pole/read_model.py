import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("cart_pole/cart_pole.xml")
data = mujoco.MjData(model)

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)

def angle_norm(theta, mode='deg'):
    if mode == 'deg':
        t = theta
    elif mode == 'rad':
        t = np.rad2deg(theta)
    else:
        raise Exception("Mode: deg or rad")
    t = t % 360 - 180
    return t
    

# 初始化：摆杆偏离竖直方向 5°
data.qpos[0] = 0.0
data.qpos[1] = np.deg2rad(185)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # 暂时施加零输入，观察系统自然运动
        data.ctrl[0] = 0.0

        mujoco.mj_step(model, data)
        
        print("theta:", angle_norm(data.qpos[1], 'rad'))

        viewer.sync()
        
        time.sleep(0.01)