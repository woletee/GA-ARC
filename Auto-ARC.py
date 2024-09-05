import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Transformation functions (rotation, flipping, color changing)
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

# Function to apply fill rule: turn black pixels surrounded by green into yellow
def apply_fill_rule(input_grid):
    output_grid = copy.deepcopy(input_grid)
    rows, cols = input_grid.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if input_grid[i, j] == 0:  # Black pixel
                # Check if surrounded by green pixels
                if input_grid[i - 1, j] == 3 and input_grid[i + 1, j] == 3 and \
                   input_grid[i, j - 1] == 3 and input_grid[i, j + 1] == 3:
                    output_grid[i, j] = 4  # Fill with yellow
    return output_grid

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
    return transformed_grid

# Generate the initial population of grids with random grid sizes
def generate_initial_population(num_individuals=10, min_size=5, max_size=10):
    population = []
    for _ in range(num_individuals):
        rows = random.randint(min_size, max_size)
        cols = random.randint(min_size, max_size)
        individual = {
            'input_grid': np.random.choice([0, 3], size=(rows, cols), p=[0.7, 0.3])  # Mostly black, some green
        }
        # Ensure each grid follows the fill rule
        individual['output_grid'] = apply_fill_rule(individual['input_grid'])
        population.append(individual)
    return population

# Updated fitness function to encourage filling more pixels
def fitness(individual):
    input_grid = individual['input_grid']
    expected_output = apply_fill_rule(input_grid)
    
    # Count the number of pixels that were correctly filled
    correctly_filled_pixels = np.sum((individual['output_grid'] == expected_output) & (expected_output == 4))
    
    # Add a bonus for each correctly filled pixel to encourage more filling
    fitness_score = correctly_filled_pixels * 10  # Bonus for correct fills
    
    return fitness_score

# Select best individuals based on fitness
def select_best_individuals(population, num_best=5):
    sorted_population = sorted(population, key=lambda x: fitness(x), reverse=True)
    return sorted_population[:num_best]

# Crossover function to combine two parent grids
def crossover(parent1, parent2):
    rows1, cols1 = parent1['input_grid'].shape
    rows2, cols2 = parent2['input_grid'].shape
    
    # Crossover only if parents have the same grid size
    if rows1 == rows2 and cols1 == cols2:
        child_input = parent1['input_grid'].copy()
        crossover_point = random.randint(1, rows1 - 2)
        child_input[crossover_point:, :] = parent2['input_grid'][crossover_point:, :]
        child_output = apply_fill_rule(child_input)
        return {'input_grid': child_input, 'output_grid': child_output}
    else:
        return random.choice([parent1, parent2])  # If sizes don't match, pick one parent

# Mutate the grid by randomly changing some pixels
def mutate(individual, mutation_rate=0.1):
    grid = individual['input_grid']
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if random.random() < mutation_rate:
                grid[i, j] = random.choice([0, 3])  # Change between black and green
    individual['output_grid'] = apply_fill_rule(grid)
    return individual

# Main Genetic Algorithm Loop
def genetic_algorithm(num_generations=50, population_size=10, min_size=5, max_size=10):
    population = generate_initial_population(population_size, min_size, max_size)
    for generation in range(num_generations):
        best_individuals = select_best_individuals(population)
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(best_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    return select_best_individuals(population, num_best=1)[0]

# Visualization functions
def plot_one(ax, matrix, title):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_xticks([x - 0.5 for x in range(1 + len(matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(matrix))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_task(input_task, output_task, task_number):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(f'{task_number} Generated Task')
    plot_one(axs[0], input_task, 'Generated Input')
    plot_one(axs[1], output_task, 'Generated Output')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Generate 20 new tasks with diverse grid sizes and visualize them
new_tasks = []
for i in range(60):  # Change this from 10 to 20
    best_individual = genetic_algorithm(num_generations=50, population_size=60, min_size=5, max_size=10)
    generated_input = best_individual['input_grid']
    generated_output = best_individual['output_grid']
    plot_task(generated_input, generated_output, f"Task {i + 1}")
    new_tasks.append((generated_input, generated_output))
