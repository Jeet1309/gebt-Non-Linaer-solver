import os
import shutil
import numpy as np
import extract_data 
import modify_input 
import produce_output
import extract_data

def compute_Sn(Sij, gamma11, kappa1, kappa2, kappa3):
    """
    Compute nonlinear section stiffness matrix Sn (6x6) from 9x9 Sij matrix.
    Inputs:
    - Sij: 9x9 numpy array (section matrix)
    - gamma11, kappa1, kappa2, kappa3: deformation measures (scalars)
    Returns:
    - Inverse of Sn (6x6) if invertible, else None
    """
    Sn = np.zeros((6, 6))

    # Compute elements:
    Sn11 = Sij[0, 0] + Sij[6, 6] * kappa2**2
    Sn12 = Sij[0, 1] + Sij[1, 6] * kappa2 + Sij[6, 8] * kappa2**2 + 2 * kappa1 * (Sij[0, 4] + Sij[4, 6])
    Sn13 = (Sij[0, 2] + 2 * (Sij[0, 5] + Sij[2, 6]) * kappa2 + (Sij[3, 6] + Sij[0, 7]) * kappa3 +
            Sij[4, 6] * kappa1**2 + 3 * Sij[5, 6] * kappa2**2 + 2 * gamma11 * Sij[6, 6] * kappa2 +
            kappa1 * (Sij[1, 6] + 2 * Sij[6, 8] * kappa2) + 2 * Sij[6, 7] * kappa2 * kappa3)
    Sn14 = 0.0

    Sn22 = (Sij[1, 1] + 2 * Sij[2, 4] * kappa2 + 6 * Sij[4, 4] * kappa1**2 + Sij[8, 8] * kappa2**2 +
            2 * gamma11 * (Sij[0, 4] + Sij[4, 6] * kappa2) + 6 * kappa1 * (Sij[1, 4] + Sij[4, 8] * kappa2))
    Sn23 = (Sij[1, 2] + 2 * Sij[1, 5] * kappa2 + gamma11 * (Sij[1, 6] + 2 * Sij[4, 6] * kappa1 + 2 * Sij[6, 8] * kappa2) +
            2 * kappa1 * (Sij[2, 4] + Sij[8, 8]) * kappa2 + 3 * Sij[4, 8] * kappa1**2 + 3 * Sij[5, 8] * kappa2**2)
    Sn24 = 0.0

    Sn33 = (Sij[2, 2] + 2 * gamma11 * (Sij[6, 8] * kappa1 + 3 * Sij[5, 6] * kappa2 + 6 * Sij[5, 5] * kappa2**2) +
            2 * kappa1 * (Sij[1, 5] + 3 * Sij[5, 8]) * kappa2 + Sij[6, 6] * gamma11**2 +
            Sij[8, 8] * kappa1**2 + Sij[7, 7] * kappa3**2)
    Sn34 = 2 * Sij[3, 7] * kappa3 + 2 * Sij[7, 7] * kappa2 * kappa3
    Sn44 = Sij[3, 3] + 2 * Sij[3, 7] * kappa2 + Sij[7, 7] * kappa2**2

    # Fill 6x6 Sn matrix (zero rows/cols for γ̄22, γ̄33):
    Sn[0, 0] = Sn11
    Sn[0, 3] = Sn12
    Sn[0, 4] = Sn13
    Sn[0, 5] = Sn14

    Sn[3, 0] = Sn12
    Sn[3, 3] = Sn22
    Sn[3, 4] = Sn23
    Sn[3, 5] = Sn24

    Sn[4, 0] = Sn13
    Sn[4, 3] = Sn23
    Sn[4, 4] = Sn33
    Sn[4, 5] = Sn34

    Sn[5, 0] = Sn14
    Sn[5, 3] = Sn24
    Sn[5, 4] = Sn34
    Sn[5, 5] = Sn44

    try:
        Sn_inv = np.linalg.inv(Sn)
    except np.linalg.LinAlgError:
        print("❌ Sn is singular, cannot invert.")
        return None

    return Sn_inv

def extract_loads(output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic=False):
    """
    Extract force_moment (6 DOF) for each keypoint from the last step of the output file.
    Returns: dict {point_no: [Fx, Fy, Fz, Mx, My, Mz]}
    """
    data = extract_data.parse_all_steps_from_firststyle(
        output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic
    )
    last_step = data[nstep]
    points = last_step['points']

    load = {}
    for idx, pt in enumerate(points, start=1):
        load[idx] = pt['force_moment']
    return load

def give_k(load, K_current, members):
    """
    For each member:
    1. Average loads of kp1 and kp2.
    2. Get corresponding K matrix from K_current.
    3. Compute gamma vector = K @ avg_load.
    4. Use this to compute nonlinear Sn.
    5. Compute inverse Sn⁻¹ and update K_current.
    Returns:
        K_current (updated with new Sn⁻¹)
    """
    S_matrix = np.ones((9, 9))  # Replace this with real Sij matrix.

    for member in members:
        memb_no = member['memb_no']
        kp1 = member['kp1']
        kp2 = member['kp2']
        mate1 = member['mate1']  # material no.

        load_kp1 = np.array(load.get(kp1, [0, 0, 0, 0, 0, 0]))
        load_kp2 = np.array(load.get(kp2, [0, 0, 0, 0, 0, 0]))
        avg_load = 0.5 * (load_kp1 + load_kp2)

        # Find section matrix K for this material:
        section_matrix = None
        for mat in K_current:
            if int(mat['material_no']) == mate1:
                section_matrix = np.array(mat['matrix'])
                break
        if section_matrix is None:
            raise ValueError(f"Material {mate1} not found in K_current!")

        # Compute gamma vector:
        gamma_vector = section_matrix @ avg_load

        # Compute Sn:
        Sn_inv = compute_Sn(S_matrix, gamma_vector[0], gamma_vector[3], gamma_vector[4], gamma_vector[5])
        if Sn_inv is not None:
            # Update this material's section matrix:
            for mat in K_current:
                if int(mat['material_no']) == mate1:
                    mat['matrix'] = Sn_inv.tolist()
                    print(f"✅ Updated K matrix for Material {mate1} using Sn⁻¹")
        else:
            print(f"⚠️ Skipped update for Material {mate1} due to singular Sn.")

    return K_current


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

def extract_avg_displacement(output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic=False):
    """
    Extracts the average displacement magnitude over all points for the last time step.
    
    Parameters:
    - output_path: path to the GEBT output file.
    - nstep: number of time steps (we use the last one).
    - n_kp: number of key points.
    - n_member: number of members.
    - ndivs_per_member: divisions per member (list).
    - is_dynamic: whether the analysis is dynamic (bool).
    
    Returns:
    - u_avg: average displacement magnitude (scalar).
    """
    # Parse all time steps
    step_data = extract_data.parse_all_steps_from_firststyle(
        output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic
    )

    # Select the last time step
    last_step = step_data[nstep]
    points = last_step['points']

    total_disp = 0.0
    count = 0

    for pt in points:
        disp_vector = np.array(pt['displacement'][:3])  # [ux, uy, uz]
        magnitude = np.linalg.norm(disp_vector)  # Compute displacement magnitude
        total_disp += magnitude
        count += 1

    if count == 0:
        print("⚠️ Warning: No displacements found. Returning 0.")
        return 0.0

    u_avg = total_disp / count
    return u_avg

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
    output_subdir = os.path.join(output_dir, f"{base_name}_gebt_outputs")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)

    # Parse input file metadata
    try:
        metadata, members, point_conditions, section_matrices = extract_data.process_input_file(input_file, is_space=True)
    except:
        metadata, members, point_conditions, section_matrices = extract_data.process_input_file(input_file, is_space=False)

    n_kp = metadata["n_kp"]
    n_member = metadata["n_member"]
    nstep = metadata["nstep"]
    ndivs_per_member = [m["ndiv"] for m in members]
    is_dynamic = (metadata["analysis_flag"] != 0)

    # Initial stiffness matrix
    K_current = section_matrices
    print("🎯 Initial stiffness matrix K:\n", K_current)

    # Total force & scaling per iteration
    total_force, force_per_iteration = extract_and_compute_forces(point_conditions, num_iterations)

    # Convergence variables
    u_avg_prev = None

    for itr in range(1, num_iterations + 1):
        print(f"\n🔄 Starting Iteration {itr}/{num_iterations}")

        modified_input_path = os.path.join(input_dir, f"{base_name}_iter{itr}.dat")
        modify_input.modify_input_file_forces(input_file, modified_input_path, itr, force_per_iteration)
        modify_input.modify_section_matrix(modified_input_path, modified_input_path, K_current)

        # Run GEBT solver
        output_to_generate = os.path.join(output_subdir, f"{base_name}_iter{itr}.dat.out")
        try:
            produce_output.run_gebt_case(input_file=modified_input_path, output_file=output_to_generate)
        except Exception as e:
            print(f"❌ Error running GEBT at iteration {itr}: {e}")
            break

        # Compute average displacement from output
        u_avg = extract_avg_displacement(output_to_generate, nstep, n_kp, n_member, ndivs_per_member, is_dynamic)
        print(f"📏 Average displacement u_avg = {u_avg:.6e}")

        # Extract updated loads
        load = extract_loads(output_to_generate, nstep, n_kp, n_member, ndivs_per_member, is_dynamic)

        # Update stiffness matrix K
        K_current = give_k(load, K_current, members)
        print("🔧 Updated stiffness matrix K:\n", K_current)

        # Save current K matrix
        save_k_matrices(K_current, itr, output_dir)

        # Check for convergence
        if u_avg_prev is not None:
            delta_u = abs(u_avg - u_avg_prev)
            print(f"🔍 Change in u_avg = {delta_u:.6e}")
            if delta_u < 1e-6:
                print(f"✅ Converged at iteration {itr} with Δu_avg = {delta_u:.6e}")
                break
        u_avg_prev = u_avg

    # Generate final input file with total original force and last K
    final_input = os.path.join(output_dir, f"{base_name}_final.dat")
    modify_input.modify_input_file_forces(input_file, final_input, num_iterations, force_per_iteration)
    modify_input.modify_section_matrix(final_input, final_input, K_current)
    print("🏁 Final input file generated:", final_input)

    try:
        produce_output.run_gebt_case(input_file=final_input)
        print("✅ Final GEBT output generated.")
    except Exception as e:
        print(f"❌ Error running final GEBT case: {e}")

if __name__ == "__main__":
    input_file = "trial.dat"
    num_iterations = 100
    iteration_loop(input_file, num_iterations)