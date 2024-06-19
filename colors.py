import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_super_complicated_graph(num_nodes=100, edge_probability=0.2):
    """
    Create a super complicated graph using the Erdős-Rényi model.
    
    Parameters:
    num_nodes (int): Number of nodes in the graph.
    edge_probability (float): Probability of edge creation between any two nodes.
    
    Returns:
    G (networkx.Graph): Generated graph.
    """
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)
    return G

# Create a super complicated graph
num_nodes = 100
edge_probability = 0.2
G = create_super_complicated_graph(num_nodes=num_nodes, edge_probability=edge_probability)

num_nodes = len(G.nodes)
num_colors = 6  # Increase number of colors to add complexity

def k_color_fitness(state, G, k):
    conflicts = 0
    for (u, v) in G.edges():
        if state[u] == state[v]:
            conflicts += 1
    return -conflicts

# Randomized Hill Climbing
def random_neighbor(state, k):
    neighbor = state.copy()
    node = np.random.randint(len(state))
    neighbor[node] = np.random.randint(k)
    return neighbor

def randomized_hill_climbing(state, G, k, max_iter=1000):
    current_solution = state
    current_fitness = k_color_fitness(current_solution, G, k)
    fitness_over_time = [current_fitness]

    for _ in range(max_iter):
        neighbor = random_neighbor(current_solution, k)
        neighbor_fitness = k_color_fitness(neighbor, G, k)
        if neighbor_fitness > current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
        fitness_over_time.append(current_fitness)

    return current_solution, current_fitness, fitness_over_time

# Simulated Annealing
def simulated_annealing(state, G, k, max_iter=1000, initial_temp=200, cooling_rate=0.95):
    current_solution = state
    current_fitness = k_color_fitness(current_solution, G, k)
    best_solution = current_solution
    best_fitness = current_fitness
    temperature = initial_temp
    fitness_over_time = [current_fitness]

    for _ in range(max_iter):
        neighbor = random_neighbor(current_solution, k)
        neighbor_fitness = k_color_fitness(neighbor, G, k)
        if neighbor_fitness > best_fitness:
            best_solution = neighbor
            best_fitness = neighbor_fitness
        if neighbor_fitness > current_fitness or np.random.rand() < np.exp((neighbor_fitness - current_fitness) / temperature):
            current_solution = neighbor
            current_fitness = neighbor_fitness
        temperature *= cooling_rate
        fitness_over_time.append(current_fitness)

    return best_solution, best_fitness, fitness_over_time

# Genetic Algorithm
def initialize_population(size, num_nodes, k):
    return [np.random.randint(0, k, size=num_nodes) for _ in range(size)]

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(state, k, mutation_rate=0.2):
    for i in range(len(state)):
        if np.random.rand() < mutation_rate:
            state[i] = np.random.randint(k)
    return state

def tournament_selection(population, fitnesses, k=3):
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), k)
        best_candidate = max(candidates, key=lambda idx: fitnesses[idx])
        selected.append(population[best_candidate])
    return selected

def genetic_algorithm(G, k, num_nodes, pop_size=50, generations=500, mutation_rate=0.2):  # Reduced population size
    population = initialize_population(pop_size, num_nodes, k)
    best_solution = None
    best_fitness = -float('inf')
    fitness_over_time = []

    for gen in range(generations):
        fitnesses = [k_color_fitness(individual, G, k) for individual in population]
        best_index = np.argmax(fitnesses)
        if fitnesses[best_index] > best_fitness:
            best_fitness = fitnesses[best_index]
            best_solution = population[best_index]
        fitness_over_time.append(best_fitness)

        selected_population = tournament_selection(population, fitnesses)
        new_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, k, mutation_rate))
            new_population.append(mutate(child2, k, mutation_rate))

        population = new_population

    return best_solution, best_fitness, fitness_over_time

# Parameters
iterations = 10000
population_size = 50  # Reduced population size
generations = 500
mutation_rate = 0.1  # Adjusted mutation rate
initial_temperature = 100  # Increased initial temperature
cooling_rate = 0.98  # Adjusted cooling rate

# Initial state
initial_state = np.random.randint(0, num_colors, size=num_nodes)

# Run Randomized Hill Climbing
best_solution_rhc, best_fitness_rhc, fitness_over_time_rhc = randomized_hill_climbing(initial_state, G, num_colors, max_iter=iterations)

# Run Simulated Annealing
best_solution_sa, best_fitness_sa, fitness_over_time_sa = simulated_annealing(initial_state, G, num_colors, max_iter=iterations, initial_temp=initial_temperature, cooling_rate=cooling_rate)

# Run Genetic Algorithm
best_solution_ga, best_fitness_ga, fitness_over_time_ga = genetic_algorithm(G, num_colors, num_nodes, pop_size=population_size, generations=generations, mutation_rate=mutation_rate)

# Plot fitness over time
plt.figure(figsize=(6, 3))
plt.plot(fitness_over_time_rhc, label='Randomized Hill Climbing')
plt.plot(fitness_over_time_sa, label='Simulated Annealing')
plt.plot(fitness_over_time_ga, label='Genetic Algorithm')
plt.xlabel('Iterations / Generations')
plt.ylabel('Fitness')
plt.title('Fitness over Time for k-Colors Problem')
plt.legend()
plt.show()