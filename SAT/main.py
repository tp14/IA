from functools import partial
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pdb


def evaluate(attempt, clauses):
    unsatisfied_count = 0
    for clause in clauses:
        is_clause_satisfied = False
        for literal in clause:
            variable, is_negated = abs(literal), literal < 0
            if attempt[variable - 1] != is_negated:
                is_clause_satisfied = True
                break
        if not is_clause_satisfied:
            unsatisfied_count += 1
    return unsatisfied_count


def generate_neighbor(attempt):
    neighbor = attempt[:]
    num_flips = random.randint(1, 5)

    for _ in range(num_flips):
        variable_to_flip = random.randint(0, len(neighbor) - 1)
        neighbor[variable_to_flip] = not neighbor[variable_to_flip]

    return neighbor


def cooling(temp_0, temp_n, max_n, iter, schedule):
    temp_i = 0
    match schedule:
        case 0:
            temp_i = temp_0 - iter * ((temp_0 - temp_n) / max_n)
        case 1:
            temp_i = temp_0 * math.pow((temp_n / temp_0), (iter / max_n))
        case 2:
            a = ((temp_0 - temp_n) * (max_n + 1)) / max_n
            b = temp_0 - a
            temp_i = a / (iter + 1) + b
        case 3:
            a = math.log(temp_0 - temp_n) / math.log(max_n)
            temp_i = temp_0 - math.pow(iter, a)
        case 4:
            temp_i = (temp_0 - temp_n) / (
                1 + math.exp(3 * (iter - max_n * 0.5))
            ) + temp_n
        case 5:
            temp_i = (
                0.5 * (temp_0 - temp_n) * (1 + math.cos((iter * np.pi) / max_n))
                + temp_n
            )
        case 6:
            temp_i = (
                0.5 * (temp_0 - temp_n) * (1 - math.tanh((10 * iter) / max_n) - 5)
                + temp_n
            )
        case 7:
            temp_i = (temp_0 - temp_n) / math.cosh((10 * iter) / max_n) + temp_n
        case _:
            print("Cooling method not specified.")

    return temp_i


def mean_array(array):
    mean_array = [0] * len(array[0])
    for array_t in array:
        for i in range(len(array_t)):
            mean_array[i] += array_t[i]
    number_of_arrays = len(array)
    mean_array = [x / number_of_arrays for x in mean_array]
    return mean_array


def simulated_annealing(
    clauses, initial_temperature, final_temperature, max_evaluations
):
    num_variables = max(abs(literal) for clause in clauses for literal in clause)
    current_attempt = [random.choice([True, False]) for _ in range(num_variables)]
    current_result = evaluate(current_attempt, clauses)
    best_attempt = current_attempt[:]
    best_result = current_result
    iter_to_best = 0
    iter_T = 0
    SAmax = 5

    temperature = initial_temperature
    evaluations = 0

    temp_control = []
    convergence_data = []
    while evaluations < max_evaluations and temperature > final_temperature:
        while iter_T < SAmax:
            iter_T += 1
            evaluations += 1
            neighbor_attempt = generate_neighbor(current_attempt)
            neighbor_result = evaluate(neighbor_attempt, clauses)

            delta = neighbor_result - current_result

            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_attempt = neighbor_attempt
                current_result = neighbor_result

            if current_result < best_result:
                best_attempt = current_attempt[:]
                best_result = current_result
                iter_to_best = evaluations

            convergence_data.append(current_result)
        temperature = cooling(
            initial_temperature, final_temperature, max_evaluations, evaluations, 1
        )
        temp_control.append(temperature)
        iter_T = 0

    return best_result, iter_to_best, convergence_data, temp_control

def random_searching(
        clauses, max_evaluation
):
    num_variables = max(abs(literal) for clause in clauses for literal in clause)
    current_attempt = [random.choice([True, False]) for _ in range(num_variables)]
    current_result = evaluate(current_attempt, clauses)
    evaluations = 0
    iter_to_best = 0

    data = []
    while evaluations < max_evaluation:
        evaluations += 1
        new_random_attempt = [random.choice([True, False]) for _ in range(num_variables)]
        new_random_result = evaluate(new_random_attempt, clauses)

        if new_random_result < current_result:
            current_attempt = new_random_attempt
            current_result = new_random_result
            iter_to_best = evaluations
        
        data.append(current_result)

    return current_result, iter_to_best, data

# Test
initial_temperature = 100
final_temperature = 0.00001
max_evaluations = 250000

def run(clauses, _):
    return simulated_annealing(
        clauses, initial_temperature, final_temperature, max_evaluations
    )

def run_random_search(clauses, _):
    return random_searching(clauses, max_evaluations)


if __name__ == "__main__":
    # Leitura da instância do problema a partir do arquivo "uf##-01.cnf"
    files = ["250"]
    convergence = []
    temperature = []

    for i in range(len(files)):
        with open("uf" + files[i] + "-01.cnf", "r") as file:
            lines = file.readlines()

        clauses = []
        for line in lines:
            if line.startswith("%"):
                break
            if line.startswith("c") or line.startswith("p"):
                continue
            clause = [int(x) for x in line.strip().split()[:-1]]
            clauses.append(clause)

        num_runs = 20
        results = []
        results_rs = []
        from multiprocessing.pool import Pool
        from tqdm import tqdm

        with Pool() as pool:
            # for _ in range(num_runs):

            results = list(
                tqdm(
                    pool.imap_unordered(partial(run, clauses), range(num_runs), chunksize=1),
                    total=num_runs,
                )
            )
            '''
            results_rs = list(
                tqdm(
                    pool.imap_unordered(partial(run_random_search, clauses), range(num_runs), chunksize=1),
                    total=num_runs,
                )
            )
            '''
        
        #pdb.set_trace()
            # best_result, result, convergence_data, temperature_data = simulated_annealing(
            # clauses, initial_temperature, final_temperature, max_evaluations
            # )

        # results = np.array(results)
        convergence = list(map(lambda x: x[2], results))
        temperature = list(map(lambda x: x[3], results))
        #ramdom_search = list(map(lambda x: x[2], results_rs))

        print("----------")
        print(list(map(lambda x: x[0], results)))
        print("----------")

       # print("----------")
       # print(list(map(lambda x: x[0], results_rs)))
       # print("----------")

        results = list(map(lambda x: x[1], results))
        #results_rs = list(map(lambda x: x[1], results_rs))

        mean = np.mean(results)
        std = np.std(results)

        #mean_rs = np.mean(results_rs)
        #std_rs = np.std(results_rs)

        print("Experiment: " + files[i] + "-01:")
        print("Average results:", mean)
        print("Standard deviation:", std)
       # print("Random Searching - Average results:", mean_rs)
       # print("Random Searching - Standard deviation:", std_rs)

    mean_conv = mean_array(convergence)
    mean_temp = mean_array(temperature)
    #mean_mean_rs = mean_array(ramdom_search)

    best = list(map(min, zip(*convergence)))

    plt.figure(figsize=(12, 6))
    #plt.figure(figsize=(16, 6))
    
    # Plot the first graphic in the first subplot
    plt.subplot(1, 2, 1)
    plt.plot(mean_conv)
    plt.xlabel("Iterations")
    plt.ylabel("Unresolved clauses")
    plt.title("Simulated Annealing")
    #plt.legend()

    # Plot the second graphic in the second subplot
    plt.subplot(1, 2, 2)
    plt.plot(mean_temp)
    plt.xlabel("Iterations")
    plt.ylabel("Temperature")
    plt.title("Temperature")
    #plt.legend()

    # Adjust layout and display the combined figure
    '''
    plt.subplot(1, 3, 3)
    plt.plot(mean_mean_rs)
    plt.xlabel("Iterations")
    plt.ylabel("Unresolved clauses")
    plt.title("Random Search")
    '''

    plt.tight_layout()
    plt.show()

    '''
    plt.plot(mean_mean_rs)
    plt.xlabel("Iterations")
    plt.ylabel("Unresolved clauses")
    plt.title("Random Search")
    plt.show()
    '''

