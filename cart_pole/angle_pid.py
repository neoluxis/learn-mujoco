import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from envs.mujoco import MujocoEnv
from utils.plots import plot_curves

# ============================================================
# 参数配置
# ============================================================

XML_PATH = Path("cart_pole/cart_pole.xml")

SIM_TIME = 10.0

RENDER = True


