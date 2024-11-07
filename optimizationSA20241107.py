# optimizationSAoriginalversion.py
import cProfile
import numpy as np
import random
import pandas as pd
from scipy.stats import ks_2samp
from tqdm import tqdm
import time
import math
import random
from multiprocessing import Pool, Manager, Lock, Value


# Recursive Pseudo-Latin Hypercubic Sampling
def stratified_sampling(data, variables, interval_nums, intervals=""):
    # 如果变量列表为空，从当前区间随机选择一个 GWSC 的值
    if not variables:
        if len(data) > 0:
            sampled_row = data.sample(1)
            return [(intervals, sampled_row)]
        else:
            return []

    # 计算第一个变量的累积频率分布
    variable = variables[0]
    data = data.sort_values(variable)
    data[variable + "_rank"] = np.arange(len(data)) / (len(data) - 1)

    # 使用对应的区间数
    bounds = np.linspace(0, 1, interval_nums[variable] + 1)

    results = []
    # 对每个区间进行处理
    for i in range(len(bounds) - 1):
        interval_data = data[
            (data[variable + "_rank"] >= bounds[i])
            & (data[variable + "_rank"] < bounds[i + 1])
        ]
        # 递归地调用该函数
        result = stratified_sampling(
            interval_data,
            variables[1:],
            interval_nums,
            intervals + str(i + 1),
        )
        results.extend(result)
    return results


def fitness(
    data,
    individual,
    variables,
    seednum,
    original_kendallcorr_matrix,
    ks_weight,
    kdcorchange_weight,
    corrdrop_columns,
):
    interval_nums = {var: num for var, num in zip(variables, individual)}

    # 设置随机种子
    np.random.seed(seednum)
    # 对每个变量进行Optimizing Recursive Pseudo-Latin Hypercubic Sampling
    sampled_intervals = stratified_sampling(data, variables, interval_nums)

    # 重置随机种子
    np.random.seed(int(time.time()))

    # 创建sampled_data
    sampled_data = pd.concat([row for _, row in sampled_intervals])

    # 计算每个变量的 KS 检验值
    ks_pvalues = {}
    for variable in variables:
        _, ks_pvalue = ks_2samp(data[variable], sampled_data[variable])
        ks_pvalues[variable] = ks_pvalue

    # 找到这个组合中 KS 检验值最小的变量
    min_ks_pvalue = min(ks_pvalues.values())

    # 计算新生成的sampled_data相对原data的kendall相关系数变化最大的值
    sampled_kendall_matrix = sampled_data.drop(corrdrop_columns, axis=1).corr(
        method="kendall"
    )
    kdcorchange_values = abs(original_kendallcorr_matrix - sampled_kendall_matrix)
    max_kdcorchange_value = kdcorchange_values.max().max()

    # 对min_ks_value和max_kdcorchange_value加权求和，得到fitness_value
    fitness_value = (
        ks_weight * min_ks_pvalue + kdcorchange_weight * (1 - max_kdcorchange_value)
    ) / (ks_weight + kdcorchange_weight)

    return fitness_value


def generate_new_solution(
    individual,
    variables,
    sample_numlow,
    sample_numup,
    individual_low,
    individual_up,
    variable_groups,
):
    size = len(individual)
    new_individual = individual.copy()
    indices = list(range(size))  # 创建一个包含所有索引的列表
    random.shuffle(indices)  # 随机打乱索引
    for i in indices:
        # 计算位移量，位移数越大概率越小
        shift = np.random.choice(range(-3, 4), p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
        # 保证位移后的值在individual_low和individual_up之间
        if individual_low <= new_individual[i] + shift <= individual_up:
            new_individual[i] += shift
            # 检查新的抽样数目是否在sample_numlow和sample_numup的范围内
            sample_num = np.prod(new_individual)
            while not (sample_numlow <= sample_num <= sample_numup):
                # 如果不在范围内，需要判定new_individual[i]是往大了调还是小了调
                other_indices = [
                    j for j in indices if j != i
                ]  # 创建一个除了i之外的索引列表
                for j in other_indices:
                    if sample_num > sample_numup:  # 超出上限
                        if individual_low < new_individual[j] < individual_up:
                            while (
                                new_individual[j] > individual_low
                                and sample_num > sample_numup
                            ):
                                new_individual[j] -= 1
                                sample_num = np.prod(new_individual)
                            if sample_numlow <= sample_num <= sample_numup:
                                break
                    else:  # 超出下限
                        if individual_low < new_individual[j] < individual_up:
                            while (
                                new_individual[j] < individual_up
                                and sample_num < sample_numlow
                            ):
                                new_individual[j] += 1
                                sample_num = np.prod(new_individual)
                            if sample_numlow <= sample_num <= sample_numup:
                                break
                else:  # 遍历完所有的j后，如果sample_num仍然不在范围内，逐步缩小调整步长
                    step = 1
                    while not (sample_numlow <= sample_num <= sample_numup):
                        if sample_num > sample_numup:  # 超出上限
                            new_individual[i] -= step
                        else:  # 超出下限
                            new_individual[i] += step
                        sample_num = np.prod(new_individual)
    # 随机调整每个组内部的变量顺序
    for group in variable_groups:
        random.shuffle(group)

    # 随机调整组的顺序
    random.shuffle(variable_groups)

    # 将调整后的variable_groups赋值给new_variable_groups
    new_variable_groups = variable_groups

    # 根据新的变量组顺序和组内部的变量顺序来调整variables和new_individual的顺序
    new_order = [var for group in new_variable_groups for var in group]
    new_variables, new_individual = map(
        list,
        zip(
            *sorted(zip(variables, new_individual), key=lambda x: new_order.index(x[0]))
        ),
    )
    # 随机调整随机种子
    new_seednum = random.randint(0, 2147483647)
    return new_individual, new_variables, new_seednum, new_variable_groups


def generate_solution_worker(args):
    (
        local_best_individual,
        local_best_variables,
        sample_numlow,
        sample_numup,
        individual_low,
        individual_up,
        local_best_variable_groups,
        generated_solutions,
        lock,
    ) = args
    while True:
        new_individual, new_variables, new_seednum, new_variable_groups = (
            generate_new_solution(
                local_best_individual,
                local_best_variables,
                sample_numlow,
                sample_numup,
                individual_low,
                individual_up,
                local_best_variable_groups,
            )
        )
        new_solution = (
            tuple(new_individual),
            tuple(new_variables),
            new_seednum,
        )
        with lock:
            if new_solution not in generated_solutions:
                generated_solutions.append(new_solution)
                break
    return new_individual, new_variables, new_seednum, new_variable_groups


def fitness_worker(args):
    (
        data,
        new_individual,
        new_variables,
        new_seednum,
        original_kendallcorr_matrix,
        ks_weight,
        kdcorchange_weight,
        corrdrop_columns,
    ) = args
    return fitness(
        data,
        new_individual,
        new_variables,
        new_seednum,
        original_kendallcorr_matrix,
        ks_weight,
        kdcorchange_weight,
        corrdrop_columns,
    )


def simulated_annealing(
    data,
    individual,
    variables,
    variable_groups,
    T,
    T_min,
    alpha,
    sample_numlow,
    sample_numup,
    individual_low=2,  # 保证至少变量有2个区间
    individual_up=100,
    seednum=2024,
    objective_func_limit=None,
    max_iterations=10000,
    cycle_length=10,
    ks_weight=1,
    kdcorchange_weight=1,
    corrdrop_columns=["xcol", "ycol", "timecol"],
    show_progress=True,
    num_processes=1,  # 新增参数，默认值为1
):
    iteration = 0

    # 用于存储已经生成过的样本列表
    generated_solutions = []

    # 计算初始解的相关性
    original_kendallcorr_matrix = data.drop(corrdrop_columns, axis=1).corr(
        method="kendall"
    )

    # 计算初始解的适应度
    first_fitness = fitness(
        data,
        individual,
        variables,
        seednum,
        original_kendallcorr_matrix,
        ks_weight,
        kdcorchange_weight,
        corrdrop_columns,
    )

    # 创建一个列表来存储每次迭代的适应度值、individual和variables、seednum、variable_groups
    fitness_history = [first_fitness]
    individual_history = [individual]
    variables_history = [variables]
    seed_history = [seednum]
    variable_groups_history = [variable_groups]

    # 初始化全局最优解和其适应度
    global_best_individual = individual
    global_best_variables = variables
    global_best_fitness = first_fitness
    global_best_seednum = seednum
    global_best_variable_groups = variable_groups
    # 初始化局部最优解和其适应度
    local_best_individual = individual
    local_best_variables = variables
    local_best_fitness = first_fitness
    local_best_seednum = seednum
    local_best_variable_groups = variable_groups

    # 开始退火算法
    iterator = (
        tqdm(range(max_iterations), desc="Recursive LHS")
        if show_progress
        else range(max_iterations)
    )
    start_time = time.time()
    with Manager() as manager:
        lock = manager.Lock()  # 使用Manager对象来创建一个Lock对象
        generated_solutions = manager.list(generated_solutions)
        for _ in iterator:
            if T <= T_min:
                break
            with Pool(processes=num_processes) as pool:
                new_solutions = pool.map(
                    generate_solution_worker,
                    [
                        (
                            local_best_individual,
                            local_best_variables,
                            sample_numlow,
                            sample_numup,
                            individual_low,
                            individual_up,
                            local_best_variable_groups,
                            generated_solutions,
                            lock,
                        )
                        for _ in range(cycle_length)
                    ],
                )
                new_fitnesses = pool.map(
                    fitness_worker,
                    [
                        (
                            data,
                            new_individual,
                            new_variables,
                            new_seednum,
                            original_kendallcorr_matrix,
                            ks_weight,
                            kdcorchange_weight,
                            corrdrop_columns,
                        )
                        for new_individual, new_variables, new_seednum, new_variable_groups in new_solutions
                    ],
                )

            for (
                new_individual,
                new_variables,
                new_seednum,
                new_variable_groups,
            ), new_fitness in zip(new_solutions, new_fitnesses):
                delta_fitness = new_fitness - local_best_fitness
                acceptance_probability = np.exp(delta_fitness / T)

                if (
                    objective_func_limit is not None
                    and new_fitness >= objective_func_limit
                ):
                    break

                if delta_fitness >= 0 or np.random.rand() < acceptance_probability:
                    local_best_individual = new_individual
                    local_best_variables = new_variables
                    local_best_seednum = new_seednum
                    local_best_variable_groups = new_variable_groups
                    local_best_fitness = new_fitness

            if local_best_fitness > global_best_fitness:
                global_best_individual = local_best_individual
                global_best_variables = local_best_variables
                global_best_fitness = local_best_fitness
                global_best_seednum = local_best_seednum
                global_best_variable_groups = local_best_variable_groups

            # 将当前的individual、variables、fitness_value添加到历史记录中
            individual_history.append(local_best_individual)
            variables_history.append(local_best_variables)
            fitness_history.append(local_best_fitness)
            seed_history.append(local_best_seednum)
            variable_groups_history.append(local_best_variable_groups)

            # 内循环完成迭代后降低温度
            T = T * alpha
            iteration += 1
            elapsed_time = time.time() - start_time
            iterator.set_postfix(
                {
                    "Elapsed time": elapsed_time,
                    "ETA": (max_iterations - iteration) * elapsed_time / iteration,
                    "it/s": iteration / elapsed_time,
                    "Fitness": global_best_fitness,
                    "Samplenum": np.prod(local_best_individual),
                },
                refresh=True,
            )

        if iteration == max_iterations:
            print("Did not reach stopping criterion within max_iterations")

        # 创建一个新的DataFrame来存储所有的历史记录
        history = pd.DataFrame(
            {
                "Fitness values": fitness_history,
                "Individual history": individual_history,
                "Variables history": variables_history,
                "Seed history": seed_history,
                "variable groups history": variable_groups_history,
            }
        )

    # 生成最后迭代得到的最好结果
    # 设置随机种子
    np.random.seed(global_best_seednum)
    interval_nums = {
        var: num for var, num in zip(global_best_variables, global_best_individual)
    }
    # 对每个变量进行递归式拉丁超立方抽样抽样
    sampled_intervals = stratified_sampling(data, global_best_variables, interval_nums)
    # 重置随机种子
    np.random.seed(int(time.time()))
    # 创建sampled_data
    sampled_data = pd.concat([row for _, row in sampled_intervals])

    return (
        sampled_data,
        global_best_individual,
        global_best_variables,
        global_best_seednum,
        history,
    )


if __name__ == "__main__":
    csvFile = open(
        "G:/地下水时空动态驱动过程分析论文/codepart/csvsave/alldatatableGWSAnolake.csv",
        "r",
    )
    columns_to_read = [
        "xcol",
        "ycol",
        "timecol",
        "preliquid",
        "surfwater",
        "glemmerra",
        "anag",
        "presolid",
        "canopystor",
        "GWSA",
    ]
    # ReadData form the floder to test comsuming time
    df = pd.read_csv(
        csvFile,
        header=0,
        usecols=columns_to_read,
        dtype="float64",
        skip_blank_lines=True,
        keep_default_na=False,
    )

    # 使用你的数据和变量列表，建立最初的样本组合
    variables = [
        "preliquid",
        "surfwater",
        "glemmerra",
        "anag",
        "presolid",
        "canopystor",
        "xcol",
        "ycol",
        "timecol",
        "GWSA",
    ]

    variable_groups = [
        ["preliquid", "surfwater", "glemmerra", "anag", "presolid", "canopystor"],
        ["xcol", "ycol", "timecol"],
        ["GWSA"],
    ]

    # 初始化个体和随机种子
    original_individual = [2, 2, 2, 2, 2, 2, 2, 4, 4, 50]

    best_sampled_df, best_individual, best_variables, best_seednum, history = (
        simulated_annealing(
            df,
            original_individual,
            variables,
            variable_groups,
            T=1,
            T_min=0.001,
            alpha=0.95,
            sample_numlow=80000,
            sample_numup=110000,
            individual_low=2,
            individual_up=100,
            seednum=2024,
            objective_func_limit=None,
            max_iterations=200,
            cycle_length=20,
            ks_weight=1,
            kdcorchange_weight=1,
            corrdrop_columns=["xcol", "ycol", "timecol"],
            show_progress=True,
        )
    )

    import pickle

    # 打印结果
    print("Best individual:", best_individual)
    print("Best variables:", best_variables)
    print("Best seed number:", best_seednum)
    print("Fitness history:", history)

    # 将结果保存到pickle文件中
    with open("E:/Gitsave/smalltool/results20240418GWSAnolake.pkl", "wb") as f:
        pickle.dump(
            {
                "Best individual": best_individual,
                "Best variables": best_variables,
                "Best seed number": best_seednum,
                "Fitness history": history,
            },
            f,
        )

# Recursive LHS:  90%|███████████████████████████████████████████▏    | 90/100 [6:39:57<44:26, 266.64s/it, Elapsed time=2.4e+4, ETA=2.67e+3, it/s=0.00375, Fitness=0.963]
# Best individual: [2, 4, 2, 2, 6, 2, 2, 2, 2, 35]
# Best variables: ['preliquid', 'canopystor', 'glemmerra', 'presolid', 'surfwater', 'anag', 'timecol', 'ycol', 'xcol', 'GWSA']
# Best seed number: 1740155808
# Fitness history:     Fitness values               Individual history                                  Variables history  Seed history                            variable groups history
# Recursive LHS:  89%|████████████████████████▉   | 891/1000 [8:39:38<1:03:34, 34.99s/it, Elapsed time=3.12e+4, ETA=3.81e+3, it/s=0.0286, Fitness=0.96]
# Best individual: [3, 5, 2, 2, 2, 2, 2, 2, 2, 27]
# Best variables: ['timecol', 'ycol', 'xcol', 'glemmerra', 'anag', 'presolid', 'surfwater', 'canopystor', 'preliquid', 'GWSA']
# Best seed number: 1299842112
# Recursive LHS: 100%|██████████████████████████████████████████████████████████████████| 100/100 [8:03:01<00:00, 289.81s/it, Elapsed time=2.9e+4, ETA=0, it/s=0.00345, Fitness=0.95]
# Did not reach stopping criterion within max_iterations
# Best individual: [3, 2, 3, 2, 2, 4, 2, 2, 3, 31]
# Best variables: ['xcol', 'ycol', 'timecol', 'glemmerra', 'canopystor', 'anag', 'preliquid', 'surfwater', 'presolid', 'GWSA']
# Best seed number: 223885755
# Recursive LHS: 100%|██████████████████████████████████████████████████████| 100/100 [6:05:10<00:00, 219.11s/it, Elapsed time=2.19e+4, ETA=0, it/s=0.00456, Fitness=0.911]
# Did not reach stopping criterion within max_iterations
# Best individual: [4, 2, 2, 4, 2, 2, 2, 2, 6, 12]
# Best variables: ['ycol', 'xcol', 'timecol', 'GWSA', 'presolid', 'preliquid', 'surfwater', 'canopystor', 'anag', 'glemmerra']
# Best seed number: 1457941998
