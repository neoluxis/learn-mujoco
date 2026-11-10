# 起摆与串级 PID 稳摆

`swing_up_pid.py` 使用当前 `cart_pole.xml`，从下垂、静止状态起摆，
随后通过位置外环和角度内环稳定摆杆并让小车回中。无需修改原有程序或 XML。

## 运行

在项目根目录执行，默认显示 MuJoCo Viewer，运行 20 秒：

```bash
.venv/bin/python cart_pole/swing_up_pid.py
```

无界面运行并保存曲线：

```bash
.venv/bin/python cart_pole/swing_up_pid.py --headless
```

路径基于脚本位置解析，也可以从其他工作目录运行。默认输出目录为
`results/swing_up_pid/`，同目录再次运行会覆盖结果：

- `trajectory.png`：角度、位置、速度、推力、能量、目标倾角和模式曲线。
- `trajectory.npz`：对应的原始数据；`mode=0` 表示起摆，`mode=1` 表示串级稳摆。
- `summary.json`：本次参数、初态、切换时间、最终误差和稳定判定。

`--output-dir` 可指定其他保存目录。绘图需要 Matplotlib；`--no-plot` 只保存数据和摘要。

## 控制过程

1. **启动激励**：前 0.2 秒给小车 3 m/s² 的目标加速度，打破完全下垂静止状态。
2. **能量起摆**：根据摆杆能量差注入能量，叠加小车回中和速度阻尼，
   再通过当前模型的动力学将目标加速度换算为推力。能量控制阶段不累计 PID 积分。
3. **串级稳摆**：直立误差小于 15° 且角速度小于 1.5 rad/s 时，清零 PID 积分并切换。
   位置外环输出目标倾角，角度内环输出推力。目标倾角限幅 ±12°，推力限幅 ±20 N。
4. **失稳回退**：误差角超过 30° 时清零积分并返回起摆模式。

稳摆控制律采用当前关节和执行器方向：

```text
e_x       = x - x_ref
theta_ref = -clip(Kpx*e_x + Kix*integral(e_x) + Kdx*x_dot, ±12°)
force     =  clip(Kpa*(theta-theta_ref)
                  + Kia*integral(theta-theta_ref) + Kda*theta_dot, ±20 N)
```

微分项使用测量速度，避免目标倾角跳变导致微分冲击。两环都有条件积分防饱和，
角度内环饱和期间冻结位置外环积分。

原始关节角 `q=0°` 为下垂、`q=180°` 为直立。程序记录的 `theta` 是相对直立的
包络误差角，因此下垂对应 ±180°，稳摆目标是 0°；包络边界处绘图断线。

## 调参

默认位置 PID 为 `(0.16, 0.06, 0.155)`，角度 PID 为 `(90, 0.5, 25)`。
命令行参数均按 `Kp Ki Kd` 顺序：

```bash
.venv/bin/python cart_pole/swing_up_pid.py --headless \
  --position-pid 0.16 0.06 0.155 --angle-pid 90 0.5 25 --energy-gain 0.55
```

先通过 `--initial-angle 175` 或 `185` 单独检查直立附近的串级稳摆，
再用默认的 `--initial-angle 0` 验证起摆与切换。角度参数单位为度。
内环先调角度 Kp、Kd，再调外环位置 Kp、Kd，最后小幅调整积分。
起摆能量增益与回中增益相互影响，需要同时观察最大位移。
其余参数集中在文件顶部的 `Config` 中。

当前默认模型、默认初态的无界面仿真约在 3.33 秒进入串级稳摆，
最大位移约 1.061 m。默认参数针对该初态整定；修改下垂初始角、质量或行程后需要重新验证，
不保证任意初态都能起摆。模型惯量从 MuJoCo 质量矩阵读取，包含绿色球和关节 armature。

只有角度误差 <1°、位置误差 <0.02 m、小车速度 <0.03 m/s、角速度 <0.05 rad/s
连续保持 2 秒，且运行正常结束，才报告 `Stable: True`。
触及轨道两端 0.02 m 的检测边界会终止并保存失败数据，不将碰撞反弹视为成功。
无界面运行未稳定时退出码为 1。

## 回归验证

```bash
.venv/bin/python -m unittest discover -s tests -v
```
