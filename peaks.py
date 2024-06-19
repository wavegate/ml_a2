import numpy as np


def calculate_T1(bit_string):
    count = 0
    for bit in bit_string:
        if bit == 1:
            count += 1
        else:
            break
    return count


def calculate_T0(bit_string):
    count = 0
    for bit in reversed(bit_string):
        if bit == 0:
            count += 1
        else:
            break
    return count


def four_peaks_fitness(bit_string, T):
    T1 = calculate_T1(bit_string)
    T0 = calculate_T0(bit_string)
    n = len(bit_string)
    if T1 > T and T0 > T:
        return max(T1, T0) + n
    else:
        return max(T1, T0)


def random_neighbor(bit_string):
    neighbor = bit_string.copy()
    index = np.random.randint(len(bit_string))
    neighbor[index] = 1 - neighbor[index]  # Flip the bit
    return neighbor


def randomized_hill_climbing(bit_string, T, max_iter=1000):
    current_solution = bit_string
    current_fitness = four_peaks_fitness(current_solution, T)
    fitness_over_time = [current_fitness]

    for _ in range(max_iter):
        neighbor = random_neighbor(current_solution)
        neighbor_fitness = four_peaks_fitness(neighbor, T)
        if neighbor_fitness > current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
        fitness_over_time.append(current_fitness)

    return current_solution, current_fitness, fitness_over_time


def simulated_annealing(
    bit_string, T, max_iter=1000, initial_temp=100, cooling_rate=0.99
):
    current_solution = bit_string
    current_fitness = four_peaks_fitness(current_solution, T)
    best_solution = current_solution
    best_fitness = current_fitness
    temperature = initial_temp
    fitness_over_time = [current_fitness]

    for _ in range(max_iter):
        neighbor = random_neighbor(current_solution)
        neighbor_fitness = four_peaks_fitness(neighbor, T)
        if neighbor_fitness > best_fitness:
            best_solution = neighbor
            best_fitness = neighbor_fitness
        if neighbor_fitness > current_fitness or np.random.rand() < np.exp(
            (neighbor_fitness - current_fitness) / temperature
        ):
            current_solution = neighbor
            current_fitness = neighbor_fitness
        temperature *= cooling_rate
        fitness_over_time.append(current_fitness)

    return best_solution, best_fitness, fitness_over_time


def initialize_population(size, length):
    return [np.random.randint(2, size=length) for _ in range(size)]


def crossover(parent1, parent2):
    point = np.random.randint(len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


def mutate(bit_string, mutation_rate=0.01):
    for i in range(len(bit_string)):
        if np.random.rand() < mutation_rate:
            bit_string[i] = 1 - bit_string[i]
    return bit_string


def genetic_algorithm(T, length, pop_size=20, generations=100, mutation_rate=0.01):
    population = initialize_population(pop_size, length)
    best_solution = None
    best_fitness = -1
    fitness_over_time = []

    for _ in range(generations):
        fitnesses = [four_peaks_fitness(individual, T) for individual in population]
        best_index = np.argmax(fitnesses)
        if fitnesses[best_index] > best_fitness:
            best_fitness = fitnesses[best_index]
            best_solution = population[best_index]
        fitness_over_time.append(best_fitness)

        new_population = []
        for _ in range(pop_size // 2):
            parents = np.random.choice(
                pop_size, size=2, replace=False, p=np.array(fitnesses) / sum(fitnesses)
            )
            parent1, parent2 = population[parents[0]], population[parents[1]]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population

    return best_solution, best_fitness, fitness_over_time


import matplotlib.pyplot as plt

# Parameters
bit_string_length = 20
T = 5
iterations = 1000
population_size = 20
generations = 100
mutation_rate = 0.01
initial_temperature = 100
cooling_rate = 0.99

# Initial bit string
initial_bit_string = np.random.randint(2, size=bit_string_length)

# Run Randomized Hill Climbing
best_solution_rhc, best_fitness_rhc, fitness_over_time_rhc = randomized_hill_climbing(
    initial_bit_string, T, max_iter=iterations
)

# Run Simulated Annealing
best_solution_sa, best_fitness_sa, fitness_over_time_sa = simulated_annealing(
    initial_bit_string,
    T,
    max_iter=iterations,
    initial_temp=initial_temperature,
    cooling_rate=cooling_rate,
)

# Run Genetic Algorithm
best_solution_ga, best_fitness_ga, fitness_over_time_ga = genetic_algorithm(
    T,
    bit_string_length,
    pop_size=population_size,
    generations=generations,
    mutation_rate=mutation_rate,
)

# Plot fitness over time
plt.figure(figsize=(6, 3))
plt.plot(fitness_over_time_rhc, label="Randomized Hill Climbing")
plt.plot(fitness_over_time_sa, label="Simulated Annealing")
plt.plot(fitness_over_time_ga, label="Genetic Algorithm")
plt.xlabel("Iterations / Generations")
plt.ylabel("Fitness")
plt.title("Fitness over Time for 4-Peaks Problem")
plt.legend()
plt.show()
