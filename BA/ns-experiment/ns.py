import deepxde as dde
import numpy as np
import matplotlib as plt
import torch
#dde.backend.set_default_backend('pytorch')

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

x_min, x_max = 0, 0.06
y_min, y_max = 0, 0.24
t_min, t_max = 0, 2

# Spatial domain:
space_domain = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
# Time domain: 
time_domain = dde.geometry.TimeDomain(t_min, t_max)
# Spatio-temporal domain
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

def parabolic(x):
    x_in = x[:,0:1]
    return 277.78 * x_in * (0.06 - x_in)
    #return 0.25

def inlet(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0) # y = 0

def bot(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0) # x = 0

def top(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.06) # x = 0.06


# BC for velocity in x-direction
u_bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, top, component=0)
u_bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, bot, component=0)
#u_inlet= dde.icbc.DirichletBC(geomtime, parabolic, inlet, component=0)

# BC for velocity in y-direction
v_bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, top, component=1)
v_bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, bot, component=1)
v_inlet= dde.icbc.DirichletBC(geomtime, parabolic, inlet, component=1)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [v_inlet,
     u_bc1,
     u_bc2,
     v_bc1,
     v_bc2,
     ],
    num_domain=1500,
    num_boundary=500,
)

layer_size = [3] + [50] * 4 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
model.compile("adam", lr=1.0e-3)
losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

#rar
for i in range(60):
    X = geomtime.random_points(1000)
    err_eq = np.abs(model.predict(X, operator=pde))[0]

    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    err_eq = torch.tensor(err_eq)
    x_ids = dde.backend.to_numpy(torch.topk(err_eq,10, dim=0)[1])

    for elem in x_ids:
        print("Adding new point:", X[elem])
        data.add_anchors(X[elem])
    
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(
        iterations=1000, 
        disregard_previous_best=True, 
        callbacks=[early_stopping],
        model_save_path="models/ns"
    )
