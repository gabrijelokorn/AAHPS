import itertools
import copy
from random import randint, random
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

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
"""
test_point = np.array([0.5] * n)
for p in problems:
    print(p.name)
    print("Bounds from ", p.xl, " to ", p.xu, ".")
    print(p.evaluate(test_point))
    print(sum(p.evaluate(test_point)))
"""

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

def crow_search(df, apf=lambda c,j:random(),
                fl=lambda c,i,mi:2*np.exp(-(i/mi)*2*np.pi),
                num_crows=100, max_iter=1000):
    """ Crow search
        df:        optimization problem
        apf:       awareness probability function
        fl:        flight length function
        num_crows: number of crows (agents)
        max_iter:  maximum number of iterations
        returns:   lowest value and position of it
    """
    crow = (np.random.uniform(size=(num_crows,df.xl.shape[0]))-df.xl)/(df.xu-df.xl)
    m, vm = copy.deepcopy(crow), df.evaluate(crow)
    for _ in range(max_iter):
        for i in range(num_crows):
            j = randint(0,num_crows-1)
            if random() >= apf(crow,j):
                crow[i] += random()*fl(crow,i,max_iter)*(m[j]-crow[i])
            else:
                crow[i] = (np.random.uniform(size=df.xl.shape[0])-df.xl)/(df.xu-df.xl)
            crow[i] = np.clip(df.xl,crow[i],df.xu)
        r = df.evaluate(crow)
        ind = vm>r
        vm[ind], m[ind] = r[ind], crow[ind]
    ind = np.argmin(vm)
    return vm[ind], m[ind]

def first_descent(x,df,neigh_size=100,max_iter=100,step=0.1,tol=10e-6):
    vx = df.evaluate(x)
    for i in range(max_iter):
        it = 0
        for j in range(neigh_size):
            nx = np.clip(df.xl, x+np.random.uniform(-step, step, size=df.xl.shape[0]), df.xu)
            vnx = df.evaluate(nx)
            if vnx<vx:
                it += 1
                break
        if it==neigh_size:
            return nx, vnx
        else:
            x,vx = nx,vnx
    return vx, x

def shake(x,k,k_max,df):
    r = (np.random.uniform(size=df.xl.shape[0])-df.xl)/(df.xu-df.xl)/(k_max-k+1)
    return np.clip(df.xl,x+r,df.xu)

def variable_neighborhood_search(df, shake=shake, k_max=5, max_iter=10):
    """ Variable neighborhood search
        df:       optimization problem
        shake:    shake function
        k_max:    number of neighborhoods
        max_iter: number of iterations
    """
    x = (np.random.uniform(size=df.xl.shape[0])-df.xl)/(df.xu-df.xl)
    v = df.evaluate(x)
    for i in range(max_iter):
        k = 1
        while k <= k_max:
            x_sh = shake(x,k,k_max,df)
            v_ls, x_ls = first_descent(x_sh,df)
            if v_ls<v:
                x,v = x_ls,v_ls
                k = 1
            else:
                k += 1
    return v,x

def pareto(df):
    sum_df = copy.deepcopy(df)
    sum_df_eval = sum_df.evaluate
    sum_df.evaluate = lambda x: np.sum(sum_df_eval(x),axis=-1)
    return sum_df

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
        r, pos = opt_solver(pareto(df), **v)
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
                start = timer()
                r,c,v = meta_grid_search(func, p, **parameters)
                print(f"Problem {p.name()} with method {func.__name__} with parameters {v} at {c} with value {r} in {timer()-start} s.")
                file.write("\t".join([str(n) for n in c]) + "\n ")

if __name__ == "__main__":
    evaluate_programs([])
