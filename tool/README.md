# 🔧 GEBT non-Linear Processing & Visualization Tools

This repository provides Python tools to parse, modify, and visualize NASA GEBT (Geometrically Exact Beam Theory) solver data. It enables:

- **Automated Iterative Analysis** (modifying stiffness matrices based on displacement adn forces)
- **Deformation Visualization** 
- **Eigenmode Visualization and non-Linear solver**  (solver is yet to be implemented)

---

## 📦 Repository Structure
```php-template
/
├── extract_data.py # GEBT output parser (steady, dynamic, eigen)
├── iteration_run.py # Iterative input modification + GEBT run
├── plot_animation.py # Deformation and force/moment visualization
├── examples/ # Sample GEBT input files
├── output_iteration/ # Auto-generated per iteration
│ ├── input_iteration/ # Modified GEBT input files
│ └── output_iteration/ # Corresponding GEBT output files
```
---

## ✅ Requirements

```bash
pip install numpy matplotlib
```
## 🚀 Usage Guide
### 0. Clone this repo 
```bash
git clone https://github.com/Jeet1309/gebt-Non-Linaer-solver
cd gebt-Non-Linaer-solver
cd tools
```
### 1. Parsing and Viewing GEBT Output
make sure you save this code in tolls directory itself or change the import file to 
```python 
import path.to.extract_data(or anyother file you need)
```

```python
import extract_data

input_file = "examples/cantilever.dat"
output_file = "gebt_outputs/cantilever.dat.out"

metadata, members, point_conditions, section_matrices = extract_data.process_input_file(input_file, is_space=True)

steps = extract_data.parse_all_steps_from_firststyle(
    output_file,
    metadata["nstep"],
    metadata["n_kp"],
    metadata["n_member"],
    [m["ndiv"] for m in members],
    is_dynamic=(metadata["analysis_flag"] != 0)
)

extract_data.print_all_gebt_output(steps)
```
### 2. Running Iterative Load & Stiffness Update
Edit iteration_run.py and set:

```python

input_template = "examples/cantilever.dat"    # Input file path
build_dir = "gebt_solver/"                    # Path to GEBT executable
num_iterations = 5                            # Number of iterations
```

Then run:
```bash
python iteration_run.py
```
Each iteration's input/output and updated stiffness matrix will be saved in:

```php-template

output_iteration/
├── input_iteration/
├── output_iteration/
└── stiffness_iteration_<n>.npy
```
### 3. Visualizing Deformation 
Edit plot.py and set:
```python 
input_files = [f"trial_output_iteration/trial_input_iteration/trial_iter{i}.dat" for i in range(1,101)]   ## list of files you want to plot give the lcation of input and output files (you can 
output_files = [f"trial_output_iteration/trial_gebt_outputs/trial_iter{i}.dat.out" for i in range(1,101)]  ##provide a single file or multiple as shown it will still show all the steps and eigenvalues.
process_files_and_plot(input_files,output_files)
```
To animate deformation :

```bash
python plot.py 
```
- Press 'n' to step through simulation steps or eigenmodes.



## 🎯 Features
- 📊 Fully parsed GEBT output (static/dynamic/eigen)

- 🔁 Automatic force scaling and stiffness matrix update

- 🎥 Interactive deformation & eigenmode animations

- 📝 Stiffness history saved for debugging and analysis

⚠️ Notes
GEBT solver (gebt) is NOT included.
You must have GEBT installed and compiled separately.

This toolkit only automates input modification, output parsing, and visualization around GEBT.

