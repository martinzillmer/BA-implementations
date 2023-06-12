import deepxde as dde
import numpy as np

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
    exp = dde.backend.pytorch.exp
else:
    from deepxde.backend import tf
    sin = tf.sin
    exp = tf.exp


rho = 1060  # density
eta = 0.005 # viscosity 

def pde(x, y):
    u = y[:, 0:1] # x-coord
    v = y[:, 1:2] # y-coord
    p = y[:, 2:3] # pressure
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    du_t = dde.grad.jacobian(y, x, i=0, j=2)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dv_t = dde.grad.jacobian(y, x, i=1, j=2)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    continuity = du_x + dv_y
    x_momentum = du_t + u*du_x + v*du_y + 1/rho*dp_x - eta*(du_xx + du_yy)
    y_momentum = dv_t + u*dv_x + v*dv_y + 1/rho*dp_y - eta*(dv_xx + dv_yy)
    return [continuity, x_momentum, y_momentum]