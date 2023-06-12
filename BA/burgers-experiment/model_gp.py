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
dim_num_dense_layers = Integer(low=1, high=10, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
dim_activation = Categorical(categories=["sin", "sigmoid", "tanh"], name="activation")
# dim_rar_iter = Integer(low, 10, 100, name=num_rar_iter)

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    #dim_rar,
]

# set the default_parameters
default_parameters = [1e-3, 3, 32, "tanh"]

# fitness function
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):

    test = Parser()
    config = test.config
    config.learning_rate = learning_rate
    config.num_dense_layers = num_dense_layers
    config.num_dense_nodes = num_dense_nodes
    config.activation = activation
    
    global ITERATION
    

    config.name = config.name + 'gp-' + str(ITERATION)
    print(config.name, 'config.name')
    print(ITERATION, 'it number')

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(config.learning_rate))
    print('num_dense_layers:', config.num_dense_layers)
    print('num_dense_nodes:', config.num_dense_nodes)
    print('activation:', config.activation)
    print()

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Data
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        [], 
        num_domain=config.num_domain,  
        #num_boundary=0, 
        #num_initial=0,
        #num_test=1000
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


