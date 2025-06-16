def modify_input_file_forces(input_file, modified_input_path, iteration, force_per_iteration):
    output_lines = []
    with open(input_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        if stripped_line.endswith('# point condition'):
            point_line = lines[i].strip()
            point_id = int(point_line.split()[0])

            output_lines.append(line)  # write '# point condition' line

            # Next line: DOFs
            i += 1
            dof_line = lines[i].strip()
            dofs = list(map(int, dof_line.split()))

            # ---- Always write the DOF line again ----
            output_lines.append(' '.join(map(str, dofs)) + '\n')

            # Next line: values
            i += 1
            value_line = lines[i].strip()
            values = list(map(float, value_line.split()))

            # Next: time_function
            i += 1
            time_function_line = lines[i].strip()

            # Next: follower
            i += 1
            follower_line = lines[i].strip()

            # ---- Modify force values if this point is in force_per_iteration ----
            if point_id in force_per_iteration:
                new_values = []
                for dof in dofs:
                    if dof in range(7, 13):  # DOFs 7-12
                        idx = dof - 7
                        new_force = force_per_iteration[point_id][idx] * iteration
                        new_values.append(f"{new_force:.6f}")
                    else:  # DOFs 1-6 remain zero
                        new_values.append("0.0")
                new_value_line = ' '.join(new_values) + '\n'
            else:
                new_value_line = value_line + '\n'  # No change if point not in force_per_iteration

            # ---- Write modified sections ----
            output_lines.append(new_value_line)
            output_lines.append(time_function_line + '\n')
            output_lines.append(follower_line + '\n')

        else:
            output_lines.append(line)
        i += 1

    # ---- Write final modified file ----
    with open(modified_input_path, 'w') as f:
        f.writelines(output_lines)
def modify_section_matrix(input_file, modified_input_path, k_current):
    output_lines = []
    with open(input_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        # Detect section matrix
        if stripped_line.endswith("# section matrix"):
            material_no = int(stripped_line.split()[0])
            output_lines.append(line)  # Keep the section matrix header line

            # Find corresponding material in k_current
            matrix_to_insert = None
            for mat in k_current:
                if mat['material_no'] == material_no:
                    matrix_to_insert = mat['matrix']
                    break

            if matrix_to_insert is not None:
                # Replace the next 6 lines with new matrix
                for row in matrix_to_insert:
                    formatted_row = ' '.join([f"{val:.10e}" for val in row]) + '\n'
                    output_lines.append(formatted_row)
                i += 6  # Skip the old matrix lines
            else:
                # No updated matrix found, keep the original 6 lines
                for j in range(6):
                    output_lines.append(lines[i + 1 + j])
                i += 6  # Skip these lines

        else:
            output_lines.append(line)

        i += 1

    # Write modified output
    with open(modified_input_path, 'w') as f:
        f.writelines(output_lines)
# ---------------------- Main Execution ---------------------- #
if __name__ == "__main__":
    input_file = "gebr_auto_input.dat"
    modified_input_path = "real_output.dat"
    iteration =2
    k_current = [
        {
            'material_no': 1,
            'matrix': [
                [1.9230769231e-08, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0689655172e-05, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.7692307692e-06, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.3076923077e-05]
            ]
        }
    ]
    point_conditions = [
        {'point': 1, 'dof': [1,2,3,4,5,6], 'value': [0,0,0,0,0,0], 'time_function': [0,0,0,0,0,0], 'follower':[0,0,0,0,0,0]},
        {'point': 2, 'dof': [7,8,9,10,11,12], 'value': [0,100000,100000,0,0,0], 'time_function':[0,1,1,0,0,0], 'follower':[0,0,0,0,0,0]}
    ]
    total_force = {
        2: [0.0, 100000.0, 50000.0, 0.0, 0.0, 0.0]
    }

    force_per_iteration = {
        2: [0.0, 20000.0, 10000.0, 0.0, 0.0, 0.0]
    }

    modify_input_file_forces(input_file, modified_input_path, iteration, force_per_iteration)
    modify_section_matrix(modified_input_path, modified_input_path, k_current)


