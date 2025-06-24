# GEBT Solver Setup Guide (Linux)

This guide explains how to set up and build the GEBT solver on a Linux system from source.

## Requirements

Before starting, make sure you have the following installed:

- **CMake (version ≥ 3.10)**
- **GNU Fortran Compiler (gfortran)**
- **Make**
- **Git**

### Install prerequisites (if not installed):

```bash
sudo apt update
sudo apt install -y cmake gfortran make git
```
## Step 1: Clone the GEBT Solver Repository
```bash
git clone https://github.com/wenbinyugroup/gebt
cd GEBC
```
## Step 2: Build the Solver
Create a build directory and compile the source using CMake:
```bash
mkdir build
cd build
cmake ..
make

```
## Step 3: Verify the Build

After compilation, the solver executable gebt (or similarly named binary) will be present in the build directory.

Run the following command to verify:
```bash
./gebt
```
If no arguments are provided, the solver may display usage information or an error regarding missing input files — this indicates that the build was successful.

## Step 4: Running a GEBT Simulation
Prepare your GEBT input file (.dat) according to the required format.

Run the solver by specifying the input file:
```bash
./gebt path/to/input_file.dat
```
Output files (e.g., .out, .ech) will be generated in the same directory.

## Notes
- The solver currently supports Linux only.

- For solver documentation, refer to the /doc directory if available.

- For developing or modifying GEBT, make sure you have gfortran version 8 or later for full compatibility.

  
## Troubleshooting

| Problem                                | Solution                                                                           |
|----------------------------------------|-----------------------------------------------------------------------------------|
| `cmake: command not found`              | Install CMake: `sudo apt install cmake`                                           |
| `gfortran: command not found`           | Install GNU Fortran: `sudo apt install gfortran`                                  |
| Permission denied running `gebt`        | Make the sudo                         |

- Once you have successfully installed and ran the GEBT solver you can go and check tools directory provided to run the non_linear solver on GEBT 
