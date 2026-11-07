
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        
    @staticmethod
    def createPID(kp, ki, kd, pid_type='pos'):
        """
        创建 PID 控制器实例。

        Parameters
        ----------
        kp : float
            比例增益。
        ki : float
            积分增益。
        kd : float
            微分增益。
        pid_type : str, optional
            PID 控制器类型，可选 'pos'（位置式）或 'inc'（增量式），默认为 'pos'。

        Returns
        -------
        PID_pos or PID_inc
            根据指定类型创建的 PID 控制器实例。
        """
        if pid_type == 'pos':
            return PID_pos(kp, ki, kd)
        elif pid_type == 'inc':
            return PID_inc(kp, ki, kd)
        else:
            raise ValueError("Invalid pid_type. Choose 'pos' or 'inc'.")
    
    

class PID_pos(PID):
    """ 
    离散位置式 PID 控制器。
    公式：y(t) = K_p * e(t) + K_i * Sigma e(t) * dt + K_d * (e(t) - e(t-1)) / dt
    """
    def __init__(self, kp, ki, kd):
        super().__init__(kp, ki, kd)
        
    def reset(self):
        super().reset()
    
    def update(self, error, dt):
        """
        根据当前误差计算控制输出。

        Returns
        -------
        output : float
            控制输出。
        """

        # 积分项
        self.integral += error * dt

        # 微分项
        derivative = (error - self.prev_error) / dt

        # PID 输出
        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        # 更新前一误差
        self.prev_error = error

        return output
    
class PID_inc(PID):
    """ 
    离散增量式 PID 控制器。
    公式：Delta y(t) = K_p * (e(t) - e(t-1)) + K_i * e(t) * dt + K_d * (e(t) - 2*e(t-1) + e(t-2)) / dt
    """
    
    def __init__(self, kp, ki, kd):
        super().__init__(kp, ki, kd)
        self.prev_prev_error = 0.0

    def reset(self):
        super().reset()
        self.prev_prev_error = 0.0

    def update(self, error, dt):
        """
        根据当前误差计算控制输出。

        Returns
        -------
        output : float
            控制输出。
        """

        # 增量式 PID 输出
        output = (
            self.kp * (error - self.prev_error)
            + self.ki * error * dt
            + self.kd * (error - 2 * self.prev_error + self.prev_prev_error) / dt
        )

        # 更新前一误差和前前误差
        self.prev_prev_error = self.prev_error
        self.prev_error = error

        return output