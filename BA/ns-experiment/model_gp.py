# Simple template to run HPO on a model defined in model.py
import deepxde as dde
import numpy as np
from config import Parser
import skopt 
from model import create_model, train_model 
from pde_transform import pde
#from tensorflow.keras import backend as K
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

ITERATION = 0

# define the HPs:
dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=8, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=400, name="num_dense_nodes")
dim_ratio = Real(low=0.6, high=0.9, name="ratio")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_ratio,
]

# set the default_parameters
default_parameters = [1e-3, 4, 50, 0.66]

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

# fitness function
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, ratio):

    test = Parser()
    config = test.config
    config.learning_rate = learning_rate
    config.num_dense_layers = num_dense_layers
    config.num_dense_nodes = num_dense_nodes
    config.ratio = ratio
    
    global ITERATION
    

    config.name = config.name + 'gp-' + str(ITERATION)
    print(config.name, 'config.name')
    print(ITERATION, 'it number')

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(config.learning_rate))
    print('num_dense_layers:', config.num_dense_layers)
    print('num_dense_nodes:', config.num_dense_nodes)
    print('ratio:', config.ratio)
    print()

    x_min, x_max = 0, 0.06
    y_min, y_max = 0, 0.24
    t_min, t_max = 0, 2

    # Spatial domain:
    space_domain = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    # Time domain: 
    time_domain = dde.geometry.TimeDomain(t_min, t_max)
    # Spatio-temporal domain
    geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)



    u_bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, top, component=0)
    u_bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, bot, component=0)
    #u_inlet= dde.icbc.DirichletBC(geomtime, parabolic, inlet, component=0)

    # BC for velocity in y-direction
    v_bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, top, component=1)
    v_bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, bot, component=1)
    v_inlet= dde.icbc.DirichletBC(geomtime, parabolic, inlet, component=1)


    nd = int(config.num_residuals * ratio)
    nb = int((config.num_residuals - nd) *7/8)
    ni = int((config.num_residuals - nd) *1/8)
    # Data
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        [u_bc1,u_bc2,v_bc1,v_bc2,v_inlet], 
        num_domain=nd,  
        num_boundary=nb, 
        num_initial=ni,
    )
    
    # Create the neural network with these hyper-parameters.
    model = create_model(config, data)
    # possibility to change where we save
    accuracy = train_model(model, config, data, geomtime)
    #print(accuracy, 'accuracy is')
        
    if np.isnan(accuracy):
        accuracy = 10 ** 5

    
    ITERATION += 1
    return accuracy
    

if __name__ == "__main__":
    n_calls = 30
    
    test = Parser()
    config = test.config

    search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls = n_calls,
                            x0=default_parameters,
                            random_state = config.seed)



    name = 'results/' + config.name + 'gp' 
    search_result.x

    skopt.dump(search_result, name + '.pkl')


