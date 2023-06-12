import deepxde as dde
import numpy as np

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
    exp = dde.backend.pytorch.exp
else:
    from deepxde.backend import tf
    sin = tf.sin
    exp = tf.exp

# IC + BC
def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    return (1 - x_in) * (1 + x_in) * (1 - exp(-t_in)) * y - sin(np.pi * x_in)

# gPINN
def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, j=0)
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    dy_tx = dde.grad.hessian(y, x, i=0, j=1)
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)

    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)
    return [
        dy_t + y * dy_x - 0.01 / np.pi * dy_xx,
        dy_tx + (dy_x * dy_x + y * dy_xx) - 0.01 / np.pi * dy_xxx,
        dy_tt + dy_t * dy_x + y * dy_tx - 0.01 / np.pi * dy_xxt,
    ]