# Auto-ARC: Automated Generation of ARC Tasks Using Genetic Algorithms

This repository contains a Python implementation of a system designed to generate new ARC (Abstraction and Reasoning Corpus) tasks using a Genetic Algorithm (GA). The generated tasks adhere to specific transformation rules, ensuring that each new task is unique while following the same underlying principles.

## Features

- **Genetic Algorithm**: The system uses a genetic algorithm to evolve and optimize transformation rules that map input grids to output grids.
- **Diverse Transformations**: Includes simple grid transformations such as rotations, flips, and color changes.
- **Uniqueness Checker**: Ensures that each generated task is unique, avoiding duplicates while maintaining adherence to the defined rules.
- **Visualization**: The generated tasks are visualized using `matplotlib`, allowing you to easily see the original task and the newly generated tasks.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Auto-ARC.git
   cd Auto-ARC
