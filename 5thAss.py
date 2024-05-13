import itertools
import numpy as np
import matplotlib.pyplot as plt

import pymoo.problems.dynamic.df as dfs

n = 75
time = 50
problems = [dfs.DF1(time = time, n_var = n),
            dfs.DF2(time = time, n_var = n),
            dfs.DF3(time = time, n_var = n),
            dfs.DF4(time = time, n_var = n),
            dfs.DF5(time = time, n_var = n),
            dfs.DF6(time = time, n_var = n),
            dfs.DF7(time = time, n_var = n),
            dfs.DF8(time = time, n_var = n),
            dfs.DF9(time = time, n_var = n),
            dfs.DF10(time = time, n_var = n),
            dfs.DF11(time = time, n_var = n),
            dfs.DF12(time = time, n_var = n),
            dfs.DF13(time = time, n_var = n),
            dfs.DF14(time = time, n_var = n)]

#Choose a sample test point (Note that this point is outside of bounds for some functions!)
test_point = np.array([0.5] * n)
for p in problems:
    print(p.name)
    print("Bounds from ", p.xl, " to ", p.xu, ".")
    print(p.evaluate(test_point))
    print(sum(p.evaluate(test_point)))

#Visualization -----------------------------------------------------
#Calculates a 2d slice of a n-dimensional space
def sum_of_paretno_functions(DF, x):
    if len(DF.xl) == 2:
        return [sum(z) for z in DF.evaluate(np.array(x))]
    else:
        xm = list((DF.xl+DF.xu)/2)
        x = [[a,b, *xm[2:]] for a, b in x]
        return [sum(z) for z in DF.evaluate(np.array(x))]

#Plots a 2d graph of a function (slice)
def plot_function(DF):
    d = 400
    x = np.linspace(DF.xl[0], DF.xu[0], d)
    y = np.linspace(DF.xl[1], DF.xu[1], d)
    X, Y = np.meshgrid(x, y)
    points = [[x, y] for x, y in zip(X.flatten(), Y.flatten())]
    Z = sum_of_paretno_functions(DF, points)
    Z = np.array(Z).reshape((d,d))
    print(Z)

    # Plotting the functions
    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    axs.contourf(X, Y, Z, levels=50, cmap='viridis')
    axs.set_title(DF.name)
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    plt.show()

def template_search(df, a=2, b=3):
    """ Optimization program template
        first argument:         optimization problem
        next keyword arguments: parameters for the method
        returns:                lowest value and position of it
    """
    print(f"Template with a={a}, b={b} tried with \"result\" {a*b}.")
    return a*b, [0]*5

def meta_grid_search(opt_solver, df, **parameters):
    """ Grid search works with problems 
        opt_solver:   optimization method solver
        df:           optimization problem
        **parameters: parameters for the optimization solver
        returns:      lowest value, position of it, best parameters found
    """
    best_r, best_params, best_pos = float("inf"), None, None
    for params in itertools.product(*parameters.values()):
        v = dict(zip(parameters.keys(), params))
        r, pos = opt_solver(df, **v)
        if r < best_r:
            best_r = r
            best_params = v
            best_pos = pos
    return best_r, best_pos, best_params

def evaluate_programs(optimizations):
    """ Evaluate 
        optimization: list of optimization methods solver and its potential parameters
        Calls optimization method solver for each function (DF01-DF14) with the given parameters and saves the results
    """
    for func,parameters in optimizations:
        with open(f"{func.__name__}.txt", "w") as file:
            for p in problems:
                r,c,v = meta_grid_search(func, p, **parameters)
                print(f"Problem {p.name()} with method {func.__name__} with parameters {v} at {c} with value {r}.")
                file.write("\t".join([str(n) for n in c]) + "\n")

if __name__ == "__main__":
    evaluate_programs([(template_search, {"a": [1,2,3], "b": [4,5,6]})])
