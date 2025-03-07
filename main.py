"""
@author: Masoud Rafiee (modified for CS446 Assignment 2)
"""
import random
import numpy as np
import myPlot as mp
import pandas as pd
import matplotlib.pyplot as plt

# --- Utility Functions ---
def getCitiesNames(bestPath, cities):
    """Get city names for all coordinates in the path, including return to start."""
    bestPathNames = []
    for coord in bestPath:
        x, y = coord[0], coord[1]
        for j in range(len(cities)):
            lat, lng = cities['lat'][j], cities['lng'][j]
            if abs(x - lat) < 1e-6 and abs(y - lng) < 1e-6:  # Floating-point tolerance
                bestPathNames.append(cities['city'][j])
                break
    return bestPathNames

def generatePopulation(n, cities_coordinates):
    """Generate a population of n random tours."""
    return [random.sample(cities_coordinates, len(cities_coordinates)) for _ in range(n)]

# --- Distance Functions ---
def manhattan(city1, city2):
    """Calculate Manhattan distance between two cities."""
    return abs(city1[0] - city2[0]) + abs(city1[1] - city2[1])

def euclidean(city1, city2):
    """Calculate Euclidean distance between two cities."""
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# --- Fitness Functions ---
def findFitness(chromosome):
    """Calculate total Manhattan distance for a tour."""
    fitness = sum(manhattan(chromosome[i], chromosome[i + 1]) for i in range(len(chromosome) - 1))
    fitness += manhattan(chromosome[-1], chromosome[0])  # Return to start
    return fitness

def findFitnessEuclidean(chromosome):
    """Calculate total Euclidean distance for a tour."""
    fitness = sum(euclidean(chromosome[i], chromosome[i + 1]) for i in range(len(chromosome) - 1))
    fitness += euclidean(chromosome[-1], chromosome[0])  # Return to start
    return fitness

def findBestFitness(population):
    """Find best fitness using Manhattan distance."""
    bestFitness, bestPath = findFitness(population[0]), population[0]
    for path in population[1:]:
        fitness = findFitness(path)
        if fitness < bestFitness:
            bestFitness, bestPath = fitness, path
    return bestFitness, bestPath

def findBestFitnessEuclidean(population):
    """Find best fitness using Euclidean distance."""
    bestFitness, bestPath = findFitnessEuclidean(population[0]), population[0]
    for path in population[1:]:
        fitness = findFitnessEuclidean(path)
        if fitness < bestFitness:
            bestFitness, bestPath = fitness, path
    return bestFitness, bestPath

# --- Selection Functions ---
def getCumulative(fitnessList):
    """Calculate cumulative probability from fitness values."""
    inverseFitness = [1 / f for f in fitnessList]
    totalFitness = sum(inverseFitness)
    probabilityCount = [i / totalFitness for i in inverseFitness]
    cumulative = []
    current = 0
    for p in probabilityCount:
        current += p
        cumulative.append(current)
    return probabilityCount, cumulative

def displayFitness(generation, use_euclidean=False):
    """Display fitness table for the population."""
    fitness = [findFitnessEuclidean(c) if use_euclidean else findFitness(c) for c in generation]
    probabilityCount, cumulative = getCumulative(fitness)
    print("\n %-6s%-11s%-16s%-13s%-10s" % ("no.", "fitness", "inverse", "probability", "cumulative"))
    print(" %-17s%-16s%-13s%-10s" % ("", "fitness", "count", "probability"))
    print("-" * 60)
    for i, (f, p, c) in enumerate(zip(fitness, probabilityCount, cumulative), 1):
        print(" %-6d%-11.1f%-16.9f%-13.6f%-12.6f" % (i, f, 1 / f, p, c))

def chooseParents(chromosomes, cumulative):
    """Select parents using Roulette Wheel selection."""
    parents = []
    for _ in range(len(chromosomes)):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r < c:
                parents.append(chromosomes[i])
                break
    return parents

def choosePairs(x):
    """Pair parents for crossover, avoiding duplicates."""
    halfSize = len(x) // 2
    a, b = x[:halfSize], x[halfSize:]
    pairs = [[a[i], b[i]] for i in range(halfSize)]
    for p in pairs:
        if p[0] == p[1]:
            for j in range(len(pairs)):
                if pairs[j][0] != p[0] and pairs[j][1] != p[0]:
                    p[0], pairs[j][0] = pairs[j][0], p[0]
                    break
    return pairs

# --- Crossover and Mutation ---
def ox_crossover(parent1, parent2):
    """Order Crossover (OX) for TSP."""
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))
    child1, child2 = [None] * size, [None] * size

    # Copy segment from parents
    for i in range(point1, point2 + 1):
        child1[i], child2[i] = parent1[i], parent2[i]

    # Fill remaining positions
    cities1 = [city for city in parent2 if city not in child1[point1:point2 + 1]]
    cities2 = [city for city in parent1 if city not in child2[point1:point2 + 1]]
    pos1, pos2 = (point2 + 1) % size, (point2 + 1) % size

    for city in cities1:
        while child1[pos1] is not None:
            pos1 = (pos1 + 1) % size
        child1[pos1] = city
        pos1 = (pos1 + 1) % size

    for city in cities2:
        while child2[pos2] is not None:
            pos2 = (pos2 + 1) % size
        child2[pos2] = city
        pos2 = (pos2 + 1) % size

    return child1, child2

def inversion_mutation(chromosome):
    """Inversion mutation: Reverse a random segment of the tour."""
    chromosome = chromosome[:]
    if len(chromosome) <= 2:
        return chromosome
    start, end = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[start:end + 1] = chromosome[start:end + 1][::-1]
    return chromosome

# --- Genetic Algorithm ---
def GA_improved(chromosomes, crossover_prob, mutation_prob, distance_func=euclidean,
                crossover_func=ox_crossover, mutation_func=inversion_mutation):
    """Improved GA with configurable functions."""
    fitness_list = [findFitnessEuclidean(c) if distance_func == euclidean else findFitness(c) for c in chromosomes]
    cumulative = getCumulative(fitness_list)[1]
    parents = chooseParents(chromosomes, cumulative)
    pairs = choosePairs(parents)

    children = []
    for p1, p2 in pairs:
        if random.random() < crossover_prob:
            c1, c2 = crossover_func(p1, p2)
        else:
            c1, c2 = p1[:], p2[:]
        c1 = mutation_func(c1) if random.random() < mutation_prob else c1
        c2 = mutation_func(c2) if random.random() < mutation_prob else c2
        children.extend([c1, c2])

    return children[:len(chromosomes)]

# --- Comparison and Testing Functions ---
def compare_distance_metrics(cities_coordinates, generations=100, pop_size=60):
    """Compare Manhattan and Euclidean distance metrics."""
    pop_manhattan = generatePopulation(pop_size, cities_coordinates)
    pop_euclidean = generatePopulation(pop_size, cities_coordinates)

    best_manhattan, best_manhattan_path = float('inf'), None
    best_euclidean, best_euclidean_path = float('inf'), None
    manhattan_history, euclidean_history = [], []

    for gen in range(generations):
        # Manhattan
        pop_manhattan = GA_improved(pop_manhattan, 0.8, 0.2, distance_func=manhattan)
        m_fitness, m_path = findBestFitness(pop_manhattan)
        manhattan_history.append(m_fitness)
        if m_fitness < best_manhattan:
            best_manhattan, best_manhattan_path = m_fitness, m_path[:]

        # Euclidean
        pop_euclidean = GA_improved(pop_euclidean, 0.8, 0.2, distance_func=euclidean)
        e_fitness, e_path = findBestFitnessEuclidean(pop_euclidean)
        euclidean_history.append(e_fitness)
        if e_fitness < best_euclidean:
            best_euclidean, best_euclidean_path = e_fitness, e_path[:]

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations + 1), manhattan_history, 'r-', label='Manhattan')
    plt.plot(range(1, generations + 1), euclidean_history, 'g-', label='Euclidean')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Convergence of Distance Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('distance_metric_convergence.png')

    return (best_manhattan, best_manhattan_path), (best_euclidean, best_euclidean_path)

def test_population_sizes(cities_coordinates, sizes=[10, 20, 30, 40, 50, 60], generations=100):
    """Test impact of population sizes on GA performance."""
    results = []
    for size in sizes:
        chromosomes = generatePopulation(size, cities_coordinates)
        best_fitness, best_path = float('inf'), None
        fitness_history = []

        for _ in range(generations):
            chromosomes = GA_improved(chromosomes, 0.8, 0.2, euclidean, ox_crossover, inversion_mutation)
            fitness, path = findBestFitnessEuclidean(chromosomes)
            fitness_history.append(fitness)
            if fitness < best_fitness:
                best_fitness, best_path = fitness, path[:]

        results.append((size, best_fitness, best_path, fitness_history))

    # Plot population size vs fitness
    sizes_list = [r[0] for r in results]
    fitness_list = [r[1] for r in results]
    mp.plotNbChromosomesVsFitness(sizes_list, fitness_list)
    plt.savefig('population_size_vs_fitness.png')

    return results

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Task 1: Load and process data
        cities = pd.read_csv('canada_cities.csv')
        cities_coordinates = cities[['lat', 'lng']].values.tolist()

        # Initial population
        n_chromosomes = 60
        chromosomes = generatePopulation(n_chromosomes, cities_coordinates)
        print("Original Population Fitness (Manhattan):")
        displayFitness(chromosomes)

        # Tasks 3 & 4: Run GA with new crossover and mutation
        generations = 100
        best_gen = [c[:] for c in chromosomes]
        best_fitness, best_path = float('inf'), None
        print("\nRunning GA with Euclidean Distance, OX Crossover, and Inversion Mutation:")
        for gen in range(generations):
            best_gen = GA_improved(best_gen, 0.8, 0.2, euclidean, ox_crossover, inversion_mutation)
            fitness, path = findBestFitnessEuclidean(best_gen)
            if fitness < best_fitness:
                best_fitness, best_path = fitness, path[:]
                print(f"Generation {gen + 1}: New best fitness = {best_fitness:.2f}")

        print("\nBest Generation Fitness (Euclidean):")
        displayFitness(best_gen, use_euclidean=True)
        print(f"\nBest Fitness = {best_fitness:.2f}")
        best_path_with_return = best_path + [best_path[0]]
        best_path_names = getCitiesNames(best_path_with_return, cities)
        print("Best Path:", ' -> '.join(best_path_names))
        mp.plotWithLabels(np.array(best_path_with_return), best_path_names,
                          title="Best Path (Euclidean Distance)")
        plt.savefig('best_path.png')

        # Task 6: Compare distance metrics
        print("\nComparing Distance Metrics:")
        (m_fitness, m_path), (e_fitness, e_path) = compare_distance_metrics(cities_coordinates)
        print(f"Manhattan - Best Fitness: {m_fitness:.2f}")
        print(f"Euclidean - Best Fitness: {e_fitness:.2f}")
        m_path_with_return = m_path + [m_path[0]]
        e_path_with_return = e_path + [e_path[0]]
        m_names = getCitiesNames(m_path_with_return, cities)
        e_names = getCitiesNames(e_path_with_return, cities)
        mp.plotWithLabels(np.array(m_path_with_return), m_names, title="Best Path (Manhattan Distance)")
        plt.savefig('manhattan_path.png')
        mp.plotWithLabels(np.array(e_path_with_return), e_names, title="Best Path (Euclidean Distance)")
        plt.savefig('euclidean_path.png')

        # Task 7: Test population sizes
        print("\nTesting Population Sizes:")
        pop_results = test_population_sizes(cities_coordinates)
        for size, fitness, _, _ in pop_results:
            print(f"Population Size {size}: Best Fitness = {fitness:.2f}")

        plt.show()  # Display all plots
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()