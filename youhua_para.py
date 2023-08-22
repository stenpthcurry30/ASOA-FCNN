# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:51:49 2023

@author: DuMengLong
"""
#%%
import pandas as pd

combined_df = pd.read_csv("asoa.csv")

#%% 优化算法
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.optim as optim
from deap import base, creator, tools, algorithms
import autograd.numpy as anp
from autograd import grad
import multiprocessing as mp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def constitutive_eq_inverse(A, alpha, epsilon_dot, n, Q_act, R, T):
    epsilon_dot = anp.clip(epsilon_dot,  1e-15, 1e15)
    A = np.clip(A, 1.0e13, 4.0e15)
    alpha = np.clip(alpha, 1e-15, 0.1)
    n = np.clip(n, 1, 10)
    Q_act = np.clip(Q_act, 300000, 700000)
    sigma = (1 / alpha) * anp.arcsinh(anp.exp((anp.log(epsilon_dot) - anp.log(A) + Q_act / (R * T)) / n))
    return sigma

def obj_func(x, sigma, epsilon_dot, T):
    A, alpha, n, log_Q_act = x
    A = np.clip(A, 1.0e13, 4.0e15)
    alpha = np.clip(alpha, 1e-15, 0.1)
    n = np.clip(n, 1, 10)
    log_Q_act = np.clip(log_Q_act, np.log(300000), np.log(700000))
    Q_act = np.exp(log_Q_act)
    R = 8.314
    sigma_pred = constitutive_eq_inverse(A, alpha, epsilon_dot, n, Q_act, R, T)
    return np.sum((sigma - sigma_pred)**2)

def optimize_params(args):
    
    
    batch, idx = args
    batch = batch.copy()  # Create a copy of the DataFrame
    def array_equal(a, b):
        return np.array_equal(a, b)
    print(f"Processing batch {idx+1}")


    A_init = 3.57e15
    alpha_init = 0.00725
    n_init = 4.03
    log_Q_act_init = np.log(435360)

    x0 = np.array([A_init, alpha_init, n_init, log_Q_act_init])

    # Step 1: Nelder-Mead optimization
    res_nm = minimize(obj_func, x0, args=(batch["Stress"], batch["Strain_rate"], batch["Temperature"]), method='Nelder-Mead')
    A, alpha, n, log_Q_act = res_nm.x
    A = np.clip(A, 1.0e15, 5.0e20)
    alpha = np.clip(alpha, 1e-15, 0.1)
    n = np.clip(n, 1, 10)
    log_Q_act = np.clip(log_Q_act, np.log(300000), np.log(700000))
    Q_act = np.exp(log_Q_act)

    # Calculate error
    sigma_pred = constitutive_eq_inverse(A, alpha, batch["Strain_rate"], n, Q_act, 8.314, batch["Temperature"])
    error = np.abs(batch["Stress"] - sigma_pred)
    
    print(f"Batch {idx+1}: Nelder-Mead optimization complete")


    # Step 2: BFGS optimization for rows with error > 5
    mask = error > 5
    iteration = 0
    max_iterations = 1000  # Choose the maximum number of iterations to avoid infinite loops
    
    sigma_numpy = batch.loc[mask, "Stress"].values
    epsilon_dot_numpy = batch.loc[mask, "Strain_rate"].values
    T_numpy = batch.loc[mask, "Temperature"].values
    
    # Define obj_func_autograd function
    def obj_func_autograd(x, sigma, epsilon_dot, T):
        A, alpha, n, log_Q_act = x
        A = np.clip(A, 1.0e15, 5.0e20)
        alpha = np.clip(alpha, 1e-15, 0.1)
        n = np.clip(n, 1, 10)
        log_Q_act = np.clip(log_Q_act, np.log(300000), np.log(700000))
        Q_act = anp.exp(log_Q_act)
        R = 8.314
        sigma_pred = constitutive_eq_inverse(A, alpha, epsilon_dot, n, Q_act, R, T)
        return anp.sum((sigma - sigma_pred)**2)
    
    # Calculate the gradient of the objective function
    obj_func_grad = grad(obj_func_autograd)
    
    x0 = res_nm.x
    
    # ...
    bounds = [(1.0e15, 5.0e20), (1e-15, 0.1), (1, 10), (np.log(300000), np.log(700000))]

    while np.any(mask) and iteration < max_iterations:
        iteration += 1
        # Perform the optimization using the Divide into batches method
        res_lbfgsb = minimize(obj_func_autograd, x0, args=(sigma_numpy, epsilon_dot_numpy, T_numpy), method='L-BFGS-B', jac=obj_func_grad, bounds=bounds, options={'maxiter': 1000})
    
        A, alpha, n, log_Q_act = res_lbfgsb.x
        A = np.clip(A, 1.0e15, 5.0e20)
        alpha = np.clip(alpha, 1e-15, 0.1)
        n = np.clip(n, 1, 10)
        log_Q_act = np.clip(log_Q_act, np.log(300000), np.log(700000))
        Q_act = np.exp(log_Q_act)
        sigma_pred[mask] = constitutive_eq_inverse(A, alpha, epsilon_dot_numpy, n, Q_act, 8.314, T_numpy)
        error[mask] = np.abs(batch.loc[mask, "Stress"] - sigma_pred[mask])
    
        print(f"Iteration {iteration}, Batch {idx+1}: L-BFGS-B optimization complete")
    
        # Update the mask to check for convergence
        mask = error > 5
        sigma_numpy = batch.loc[mask, "Stress"].values
        epsilon_dot_numpy = batch.loc[mask, "Strain_rate"].values
        T_numpy = batch.loc[mask, "Temperature"].values


    # Step 3: DEAP optimization for rows with error > 1
    mask = error > 5
    if np.any(mask):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    
        toolbox = base.Toolbox()
        toolbox.register("A", np.random.uniform, 1.0e15, 5.0e20)
        toolbox.register("alpha", np.random.uniform, 0, 0.1)
        toolbox.register("n", np.random.uniform, 1, 15)
        toolbox.register("log_Q_act", np.random.uniform, np.log(350000), np.log(750000))
        toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.A, toolbox.alpha, toolbox.n, toolbox.log_Q_act), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
        def deap_obj_func(individual):
            return (obj_func(individual, batch.loc[mask, "Stress"], batch.loc[mask, "Strain_rate"], batch.loc[mask, "Temperature"]),)
    
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=1, low=[1.0e15, 1e-15, 1, np.log(350000)], up=[5.0e20, 0.1, 15, np.log(750000)])
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=1, low=[1.0e15, 1e-15, 1, np.log(350000)], up=[5.0e20, 0.1, 15, np.log(750000)], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", deap_obj_func)
    
        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1, similar=array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("max", np.max)
    
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000, stats=stats, halloffame=hof, verbose=True)
        best_ind = hof[0]
        A, alpha, n, log_Q_act = best_ind
        Q_act = np.exp(log_Q_act)
        
        print(f"Batch {idx+1}: DEAP optimization complete")

        # Update the predictions and error for the rows optimized with DEAP
        sigma_pred[mask] = constitutive_eq_inverse(A, alpha, batch.loc[mask, "Strain_rate"], n, Q_act, 8.314, batch.loc[mask, "Temperature"])
        error[mask] = np.abs(batch.loc[mask, "Stress"] - sigma_pred[mask])
        
    # Calculate error without taking the absolute value for writing into result_df
    error_no_abs = batch["Stress"] - sigma_pred
        
    # Store the optimized parameters and predictions
    batch["A_opt"] = A
    batch["alpha_opt"] = alpha
    batch["n_opt"] = n
    batch["Q_act_opt"] = Q_act
    batch["sigma_pred"] = sigma_pred
    batch["error"] = error
    batch["error"] = error_no_abs  # Store error without taking the absolute value
    
    return batch

batch_size = 128
num_batches = len(combined_df) // batch_size + int(len(combined_df) % batch_size > 0)

batches = [combined_df.iloc[i * batch_size: min((i + 1) * batch_size, len(combined_df))] for i in range(num_batches)]

result_list = []
for i, batch in enumerate(batches):
    batch_result = optimize_params((batch,i))
    result_list.append(batch_result)

result_df = pd.concat(result_list, ignore_index=True)
#%% 保存模型
result_df.to_csv("test.csv", index=False)
    
    






























