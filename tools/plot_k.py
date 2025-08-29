import os
import numpy as np
import matplotlib.pyplot as plt

def is_nonzero(val, tol=1e-20):
    return abs(val) > tol

def load_matrix(filepath):
    return np.loadtxt(filepath)

def extract_nonzero_indices(matrix):
    return [(i, j) for i in range(matrix.shape[0])
                  for j in range(matrix.shape[1])
                  if is_nonzero(matrix[i, j])]

def process_material_folder(material_path):
    iter_files = sorted([f for f in os.listdir(material_path) if f.startswith('K_iter') and f.endswith('.txt')],
                        key=lambda x: int(x.split('iter')[-1].split('.txt')[0]))
    iteration_values = {}
    iterations = []

    for file in iter_files:
        iter_num = int(file.split('iter')[-1].split('.txt')[0])
        iterations.append(iter_num)
        matrix = load_matrix(os.path.join(material_path, file))

        # Initialize dictionary keys for nonzero entries on first iteration
        if not iteration_values:
            nonzero_indices = extract_nonzero_indices(matrix)
            for i, j in nonzero_indices:
                iteration_values[(i, j)] = []

        # Append value for each tracked nonzero index
        for (i, j) in iteration_values:
            iteration_values[(i, j)].append(matrix[i, j])

    return iterations, iteration_values

def plot_material(iterations, iteration_values, material_name, save_folder):
    plt.figure(figsize=(10, 6))
    for (i, j), values in iteration_values.items():
        plt.plot(iterations, values, label=f'K[{i},{j}]')

    plt.xlabel('Iteration')
    plt.ylabel('Non-zero Value')
    plt.title(f'Non-zero K[i,j] over Iterations - {material_name}')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f'{material_name}_nonzero_plot.png'))
    plt.close()

def process_all_materials(root_folder):
    for material in os.listdir(root_folder):
        material_path = os.path.join(root_folder, material)
        if os.path.isdir(material_path):
            print(f'Processing: {material}')
            iterations, iteration_values = process_material_folder(material_path)
            plot_material(iterations, iteration_values, material, os.path.join(root_folder, 'plots'))

# Replace with your actual top-level folder path
root_folder = 'trial_output_iteration/K_matrices'
process_all_materials(root_folder)
