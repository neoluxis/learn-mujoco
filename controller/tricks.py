import numpy as np

def output_limit(controller, limit):
    """
    限幅器修饰器。

    Parameters
    ----------
    controller : 
        被修饰的控制器。
    limit : float
        输出限幅值。

    Returns
    -------
    wrapper : function
        修饰后的控制器。
    """

    def wrapper(error, dt):
        output = controller.update(error, dt)
        output = np.clip(output, -limit, limit)
        return output

    return wrapper

def deadband(controller, threshold):
    """
    死区修饰器。

    Parameters
    ----------
    controller : 
        被修饰的控制器。
    threshold : float
        死区阈值。
    """
    
    def wrapper(error, dt):
        if abs(error) < threshold:
            return 0.0
        else:
            return controller.update(error, dt)

    return wrapper

if __name__ == "__main__":
    pass