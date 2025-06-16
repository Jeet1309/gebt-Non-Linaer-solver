import os
import shutil
import numpy as np
import extract_data  # your existing parser module
import modify_input 
import produce_output
def extract_avg_displacement(output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic=False): # can be chnaged as we need 
    data = extract_data.parse_all_steps_from_firststyle(output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic)
    
    total_disp = 0.0
    count = 0

    for step in data.values():
        # --- Points ---
        points = step['points']
        for pt in points:
            disp = pt['displacement']
            total_disp += sum(map(abs, disp))  # sum abs of all DOFs
            count += len(disp)

        # --- Members ---
        members = step['members']
        for member in members:
            for segment in member:
                disp = segment['displacement']
                total_disp += sum(map(abs, disp))
                count += len(disp)

    u_avg = total_disp / count if count != 0 else 0.0
    return u_avg


def save_k_matrices(K_current, itr, output_dir):
    k_matrix_dir = os.path.join(output_dir, "K_matrices")
    os.makedirs(k_matrix_dir, exist_ok=True)

    for mat in K_current:
        material_no = mat['material_no']
        material_dir = os.path.join(k_matrix_dir, f"material{material_no}")
        os.makedirs(material_dir, exist_ok=True)
        
        k_matrix_path = os.path.join(material_dir, f"K_iter{itr}.txt")
        np.savetxt(k_matrix_path, np.array(mat['matrix']), fmt='%.6e')
        print(f"✅ Saved K matrix for Iteration {itr}, Material {material_no} at: {k_matrix_path}")

def give_k(u_avg, K_current):# can be chnage accodring to ou need 
    """Update each material stiffness matrix in K_current."""
    scale_factor = 1 - 0.1 * abs(u_avg)
    updated_k = []

    for mat in K_current:
        new_matrix = (np.array(mat['matrix']) * scale_factor).tolist()
        updated_k.append({'material_no': mat['material_no'], 'matrix': new_matrix})

    return updated_k
def extract_and_compute_forces(point_conditions, num_iterations, target_dofs=range(7, 13)):
    total_force = {}
    force_per_iteration = {}

    if num_iterations == 0:
        raise ValueError("Number of iterations must be greater than 0.")

    for cond in point_conditions:
        dofs = cond['dof']
        values = cond['value']
        point_id = cond['point']
        
        # Build force list for DOFs 7–12; if missing, fill with 0.0
        filtered_force = []
        for target_dof in target_dofs:
            if target_dof in dofs:
                index = dofs.index(target_dof)
                filtered_force.append(values[index])
            else:
                filtered_force.append(0.0)  # pad with zero if DOF missing

        if any(filtered_force):  # only store if at least one force is non-zero
            total_force[point_id] = filtered_force
            force_per_iteration[point_id] = [f / num_iterations for f in filtered_force]

    return total_force, force_per_iteration

def iteration_loop(input_file, num_iterations):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = f"{base_name}_output_iteration"
    input_dir = os.path.join(output_dir, f"{base_name}_input_iteration")
    output_subdir = os.path.join(output_dir, f"{base_name}_gebt_outputs")  # Cleaner name
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)



    # Parse metadata
    try:
        metadata, members, point_conditions, section_matrices = extract_data.process_input_file(input_file, is_space=True)
    except:
        metadata, members, point_conditions, section_matrices = extract_data.process_input_file(input_file, is_space=False)

    n_kp = metadata["n_kp"]
    n_member = metadata["n_member"]
    ndivs_per_member = [m["ndiv"] for m in members]
    is_dynamic = (metadata["analysis_flag"] != 0)
    n_mate = metadata["n_mate"]
    n_cond_pt = metadata["n_cond_pt"]

    # Get Initial K matrix
    K_current = section_matrices
    # K_current = get_initial_K(section_matrices)
    print("🎯 Initial stiffness matrix K:\n", K_current)

    # Store original total force (used in scaling)

    total_force ,force_per_iteration = extract_and_compute_forces(point_conditions,num_iterations)
    u_avg_prev = None  # Initialize previous u_avg
    for itr in range(1, num_iterations + 1):
        print(f"\n🔄 Starting Iteration {itr}/{num_iterations}")


        modified_input_path = os.path.join(input_dir, f"{base_name}_iter{itr}.dat")
        modify_input.modify_input_file_forces(input_file, modified_input_path, itr, force_per_iteration)
        modify_input.modify_section_matrix(modified_input_path, modified_input_path, K_current)
        # Run GEBT Solver here (dummy copy in this example)
    

        input_to_run = modified_input_path
        output_to_generate = os.path.join(output_subdir, f"{base_name}_iter{itr}.dat.out")

        produce_output.run_gebt_case(input_file=input_to_run, output_file=output_to_generate)

        # # Extract output data
        

        # Update stiffness matrix
        # K_current = give_k(1, K_current)
        u_avg = extract_avg_displacement(output_to_generate, metadata["nstep"],n_kp, n_member, ndivs_per_member, is_dynamic
        )
        print(f"📏 Average displacement u_avg = {u_avg:.6e}")

        K_current = give_k(u_avg, K_current)
        print("🔧 Updated stiffness matrix K:\n", K_current)

        print("🔧 Updated stiffness matrix K:\n", K_current)

        # Save K matrix
        save_k_matrices(K_current, itr, output_dir)

                # Check convergence (optional)
        if u_avg_prev is not None:
            delta_u = abs(u_avg - u_avg_prev)
            print(f"🔍 Change in u_avg = {delta_u:.6e}")
            if delta_u < 1e-6:  # Convergence threshold
                print(f"✅ Converged at iteration {itr} with Δu_avg = {delta_u:.6e}")
                break

        u_avg_prev = u_avg 

    # Final input with original total force and last K
    final_input = os.path.join(output_dir, f"{base_name}_final.dat")
    modify_input.modify_input_file_forces(input_file, final_input, num_iterations, force_per_iteration)
    modify_input.modify_section_matrix(final_input, final_input, K_current)
    print("🏁 Final input file generated:", final_input)

    produce_output.run_gebt_case(input_file=final_input)


    print("✅ Final GEBT output generated")

if __name__ == "__main__":
    input_file = "trial.dat"
    num_iterations = 100
    iteration_loop(input_file, num_iterations)
