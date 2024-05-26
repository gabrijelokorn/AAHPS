import itertools
import copy
from random import randint, random
import numpy as np
import pandas as pd
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

def genetic_algorithm(df, pop_size=750, generations=500, mutation_prob=0.0025):
    """
    Genetic algorithm
        df:             optimization problem
        pop_size:       population size
        generations:    number of generations
        mutation_prob:  probability of mutation
        returns:        lowest value and position of it
    """
    def init_population(pop_size=100, dimensionality=75):
        return (np.random.uniform(low=df.xl, high=df.xu, size=(pop_size, dimensionality)))
    
    def evaluate_point(point, df):
        return df.evaluate(point)
    
    def abs_fitness_population(value_population):
        return 1.0 / (value_population + 1e-10)

    def normalize_fitness_population(value_population):
        return abs_fitness_population(value_population) / np.sum(abs_fitness_population(value_population))
    
    def selection(population, fitness_population):
        index = np.random.choice(len(population), size=2, p=fitness_population)
        return population[index[0]], population[index[1]], fitness_population[index[0]], fitness_population[index[1]]
    
    def crossover(parent1, parent2, fit1, fit2):
        fitness_sub_pop = fit1, fit2
        fitness_sub_pop = fitness_sub_pop / np.sum(fitness_sub_pop)

        child1 = np.zeros(len(parent1))
        child2 = np.zeros(len(parent2))
        for i in range(len(parent1)):
            if np.random.rand() < fitness_sub_pop[0]:
                child1[i] = parent1[i]
            else:
                child1[i] = parent2[i]
            if np.random.rand() < fitness_sub_pop[1]:
                child2[i] = parent2[i]
            else:
                child2[i] = parent1[i]

        return child1, child2
    
    def mutation(child, mutation_prob):
        for i in range(len(child)):
            if np.random.rand() < mutation_prob:
                child[i] = np.random.uniform(low=df.xl[i], high=df.xu[i])
        return child

    population = init_population(pop_size, len(df.xl))
    
    for _ in range(generations):
        value_population = np.array([evaluate_point(point, df) for point in population])
        fitness_population = normalize_fitness_population(value_population)
        new_population = []
        for _ in range(pop_size):
            parent1, parent2, fit1, fit2 = selection(population, fitness_population)
            child1, child2 = crossover(parent1, parent2, fit1, fit2)
            child1 = mutation(child1, mutation_prob)
            child2 = mutation(child2, mutation_prob)
            new_population.append(child1)
            new_population.append(child2)
        population = np.array(new_population)

    value_population = np.array([evaluate_point(point, df) for point in population])    
    best_index = np.argmin(value_population)
    return value_population[best_index], population[best_index]

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
    crow = df.xl+np.random.uniform(size=(num_crows,df.xl.shape[0]))*(df.xu-df.xl)
    m, vm = copy.deepcopy(crow), df.evaluate(crow)
    for _ in range(max_iter):
        for i in range(num_crows):
            j = randint(0,num_crows-1)
            if random() >= apf(crow,j):
                crow[i] += random()*fl(crow,i,max_iter)*(m[j]-crow[i])
            else:
                crow[i] =  df.xl+(np.random.uniform(size=df.xl.shape[0]))*(df.xu-df.xl)
            crow[i] = np.clip(df.xl,crow[i],df.xu)
        r = df.evaluate(crow)
        ind = vm>r
        vm[ind], m[ind] = r[ind], crow[ind]
    ind = np.argmin(vm)
    return vm[ind], m[ind]

def first_descent(x,df,neigh_size=100,max_iter=100,step=0.1):
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
    return x, vx

def shake(x,k,k_max,df,alpha=0.01):
    l = (df.xu-df.xl)/k_max*alpha
    up = np.clip(df.xl, x+l,df.xu)
    down = np.clip(df.xl, x-l,df.xu)
    r = down+(np.random.uniform(size=df.xl.shape[0]))*(up-down)
    return r

def variable_neighborhood_search(df, shake=shake, k_max=5, max_iter=10):
    """ Variable neighborhood search
        df:       optimization problem
        shake:    shake function
        k_max:    number of neighborhoods
        max_iter: number of iterations
    """
    x = df.xl+(np.random.uniform(size=df.xl.shape[0]))*(df.xu-df.xl)
    v = df.evaluate(x)
    for i in range(max_iter):
        k = 1
        while k <= k_max:
            x_sh = shake(x,k,k_max,df)
            x_ls, v_ls = first_descent(x_sh,df)
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
    res = pd.DataFrame(data=np.zeros((len(problems),1)),
                       columns = ['name'])
    res['name']=[p.name() for p in problems]
    for func,parameters in optimizations:
        with open(f"{func.__name__}.txt", "w") as file:
            rs,ts = [],[]
            for p in problems:
                start = timer()
                r,c,v = meta_grid_search(func, p, **parameters)
                t = timer()-start
                print(f"Problem {p.name()} with method {func.__name__} with parameters {v} at {c} with value {r} in {t} s.")
                file.write("\t".join([str(n) for n in c]) + "\n ")
                rs += [r]
                ts += [t]
            res[func.__name__]=rs
            res[func.__name__+' time']=ts
    return res


if __name__ == "__main__":
    results = evaluate_programs([
        # (crow_search, {}),
        # (variable_neighborhood_search, {}),
        # (genetic_algorithm, {}),
        ])
    print(results.rename(columns=lambda x: x.replace('_',' ')))
