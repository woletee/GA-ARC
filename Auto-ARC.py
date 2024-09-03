import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Simple operations for mutation
def rotate_90(grid):
    return np.rot90(grid)

def flip_horizontal(grid):
    return np.fliplr(grid)

def flip_vertical(grid):
    return np.flipud(grid)

def change_color(grid, old_color, new_color):
    new_grid = copy.deepcopy(grid)
    new_grid[grid == old_color] = new_color
    return new_grid

# Define initial ARC task
input_task = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

output_task = np.array([
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2]
])

# Genetic Algorithm Functions
def apply_transformation(grid, transformations):
    transformed_grid = copy.deepcopy(grid)
    for transformation in transformations:
        if transformation == 'rotate_90':
            transformed_grid = rotate_90(transformed_grid)
        elif transformation == 'flip_horizontal':
            transformed_grid = flip_horizontal(transformed_grid)
        elif transformation == 'flip_vertical':
            transformed_grid = flip_vertical(transformed_grid)
        elif isinstance(transformation, tuple) and transformation[0] == 'change_color':
            transformed_grid = change_color(transformed_grid, transformation[1], transformation[2])
    return transformed_grid

def generate_initial_population(input_task, num_individuals=10):
    population = []
    for _ in range(num_individuals):
        individual = {
            'transformations': random.sample(['rotate_90', 'flip_horizontal', 'flip_vertical', 
                                              ('change_color', 1, 2)], k=random.randint(1, 3))
        }
        population.append(individual)
    return population

def fitness(individual, input_task, output_task):
    transformed_input = apply_transformation(input_task, individual['transformations'])
    return np.sum(transformed_input == output_task)

def select_best_individuals(population, input_task, output_task, num_best=5):
    sorted_population = sorted(population, key=lambda x: fitness(x, input_task, output_task), reverse=True)
    return sorted_population[:num_best]

def crossover(parent1, parent2):
    combined_transformations = parent1['transformations'] + parent2['transformations']
    max_k = min(len(combined_transformations), 3)
    k = min(random.randint(1, max_k), len(combined_transformations))
    child = {
        'transformations': random.sample(combined_transformations, k=k)
    }
    return child

def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        random_transformation = random.choice(['rotate_90', 'flip_horizontal', 'flip_vertical', 
                                               ('change_color', 1, 2)])
        if random.random() < 0.5:
            individual['transformations'].append(random_transformation)
        else:
            if individual['transformations']:
                individual['transformations'].remove(random.choice(individual['transformations']))
    return individual

# Main GA Loop
def genetic_algorithm(input_task, output_task, num_generations=50, population_size=26):
    population = generate_initial_population(input_task, population_size)
    for _ in range(num_generations):
        best_individuals = select_best_individuals(population, input_task, output_task)
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(best_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    return select_best_individuals(population, input_task, output_task, num_best=1)[0]

# Function to check if the generated task is unique
def is_unique_task(new_task, existing_tasks):
    for task in existing_tasks:
        if np.array_equal(new_task[0], task[0]) and np.array_equal(new_task[1], task[1]):
            return False
    return True

# Visualization functions
def plot_one(ax, matrix, title):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_task(input_task, output_task, generated_input, generated_output, task_number):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    fig.suptitle(f'{task_number} Generated Task')
    plot_one(axs[0, 0], input_task, 'Original Input')
    plot_one(axs[0, 1], output_task, 'Original Output')
    plot_one(axs[1, 0], generated_input, 'Generated Input')
    plot_one(axs[1, 1], generated_output, 'Generated Output')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Generate 10 new tasks and visualize them
new_tasks = []
for i in range(26):
    while True:  # Loop until a unique task is generated
        best_individual = genetic_algorithm(input_task, output_task)
        new_input = np.random.randint(0, 3, size=input_task.shape)
        new_output = apply_transformation(new_input, best_individual['transformations'])
        new_task = (new_input, new_output)
        if is_unique_task(new_task, new_tasks):
            break  # If unique, exit the loop
    plot_task(input_task, output_task, new_input, new_output, f"{i+1}st")
    new_tasks.append(new_task)
