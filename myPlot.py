import numpy as np
import matplotlib.pyplot as plt

def plotWithLabels(data, labels, title="Best Path Found by Genetic Algorithm"):
    """
    Plot the path between cities with city labels.

    Parameters:
    - data (numpy.ndarray): Array of city coordinates [lat, lng]
    - labels (list): List of city names corresponding to the coordinates
    - title (str): Custom title for the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 1], data[:, 0], c='blue', s=50)  # lng (x), lat (y)
    plt.plot(data[:, 1], data[:, 0], 'r-', linewidth=1)
    for i, label in enumerate(labels):
        plt.annotate(label, (data[i, 1], data[i, 0]), textcoords="offset points",
                     xytext=(0, 10), ha='center')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.tight_layout()

def plotNbChromosomesVsFitness(population_sizes, best_fitness_list):
    """
    Plot population size vs best fitness.

    Parameters:
    - population_sizes (list): List of population sizes
    - best_fitness_list (list): List of best fitness values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(population_sizes, best_fitness_list, 'bo-', linewidth=2, markersize=8)
    plt.title('Impact of Population Size on Best Fitness')
    plt.xlabel('Population Size')
    plt.ylabel('Best Fitness (Lower is Better)')
    plt.grid(True)
    plt.tight_layout()