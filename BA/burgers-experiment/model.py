import deepxde as dde
import torch
import numpy as np
import time
import skopt
from pde_transform import output_transform, pde


def gen_testdata():
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def create_model(config, data):
        
    # Net
    net = dde.maps.FNN(
        [2] + [config.num_dense_nodes] * config.num_dense_layers + [1],
        config.activation,
        "Glorot normal",
    )

    # Apply BC-IC
    net.apply_output_transform(output_transform)

    # Model
    model = dde.Model(data, net)
    
    # Compile
    model.compile("adam", lr=config.learning_rate, loss_weights=[1, 0.0001, 0.0001])
    return model

def train_model(model, config, data, geomtime):
    
    learning_rate = config.learning_rate
    ta = time.time()
    # Train
    losshistory, _ = model.train(iterations=config.init_iterations)

    x_true, y_true = gen_testdata()
    y_pred = model.predict(x_true)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

    #rar
    for i in range(config.rar_loops):
        X = geomtime.random_points(config.random_points)
        err_eq = np.abs(model.predict(X, operator=pde))[0]

        err = np.mean(err_eq)
        print("Mean residual: %.3e" % (err))

        err_eq = torch.tensor(err_eq)
        x_ids = dde.backend.to_numpy(torch.topk(err_eq, config.add_topk, dim=0)[1])

        for elem in x_ids:
            print("Adding new point:", X[elem])
            data.add_anchors(X[elem])
        
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
        model.compile("adam", lr=learning_rate, loss_weights=[1, 0.0001, 0.0001])
        
        if i == config.rar_loops - 1:
            losshistory, _ = model.train(
                iterations=config.rar_iterations, disregard_previous_best=True, callbacks=[early_stopping], model_save_path="models/burgers"+config.name
            )
        else:
            losshistory, _ = model.train(
                iterations=config.rar_iterations, disregard_previous_best=True, callbacks=[early_stopping]
            )

        X, y_true = gen_testdata()
        y_pred = model.predict(X)
        torch.cuda.empty_cache()
        print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    texec = time.time() - ta



    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()

    skopt.dump(train, "results/" + config.name + "train.pkl")
    skopt.dump(test, "results/" + config.name + "test.pkl")
    skopt.dump(metric, "results/" + config.name + "metric.pkl")
    skopt.dump(texec, "results/" + config.name + "texec.pkl")

    
    # Test loss
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()

    # Smallest error
    error = test.min()

    return error

