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

# for p in problems:
#     plot_function(p)

# Optimization program1
def optimization_program1(df, parameters):
    print("Optimization program1: solving ", df.name, " with parameters: ", parameters)
    return None
# Optimization program2
def optimization_program2(df, parameters):
    print("Optimization program2: solving ", df.name, " with parameters: ", parameters)
    return None
# Optimization program3
def optimization_program3(df, parameters):
    print("Optimization program3: solving ", df.name, " with parameters: ", parameters)
    return None
# Optimization program4
def optimization_program4(df, parameters):
    print("Optimization program4: solving ", df.name, " with parameters: ", parameters)
    return None

# Grid search works with problems 
# opt_solver: optimization method solver
# df: df
# **parameters: dictionary of parameters for each optimization methodsolver
# 
def meta_grid_search(optimization_program, df, **parameters):
    for (k, v) in parameters.items():
        optimization_program(df, v)
    return None

# Evaluate 
# opt_solvers: list of optimization methods solver
# parameters: list of parameters for each optimization method solver
# Calls optimization method solver for each function (DF01-DF14) with the given parameters
def evaluate_programs(optimization_programs, parameters):
    for func in optimization_programs:
        for p in problems: 
            meta_grid_search(func, p, a=1, b=2, c=3)

    return None

def main():
    evaluate_programs([optimization_program1, optimization_program2, optimization_program3, optimization_program4], [None, None, None, None])

if __name__ == "__main__":
    main()