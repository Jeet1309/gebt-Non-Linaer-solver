import os
import shutil
import numpy as np
from collections import defaultdict
import extract_data 
import modify_input 
import copy
import produce_output
import extract_data
from test import Crossection_9x9

def compute_Sn( gamma11, kappa1, kappa2, kappa3):
    """
    Compute nonlinear section stiffness matrix Sn (6x6) from 9x9 Sij matrix.
    Inputs:
    - Sij: 9x9 numpy array (section matrix)
    - gamma11, kappa1, kappa2, kappa3: deformation measures (scalars)
    Returns:
    - Inverse of Sn (6x6) if invertible, else None
    """
    b = 0.1
    Sij = Crossection_9x9(kappa1,b)
    Sn = np.zeros((4, 4))

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
    Sn[0, 1] = Sn12
    Sn[0, 2] = Sn13
    Sn[0, 3] = Sn14

    Sn[1, 0] = Sn12
    Sn[1, 1] = Sn22
    Sn[1, 2] = Sn23
    Sn[1, 3] = Sn24

    Sn[2, 0] = Sn13
    Sn[2, 1] = Sn23
    Sn[2, 2] = Sn33
    Sn[2, 3] = Sn34

    Sn[3, 0] = Sn14
    Sn[3, 1] = Sn24
    Sn[3, 2] = Sn34
    Sn[3, 3] = Sn44

    try:
        Sn_inv = np.linalg.inv(Sn)
    except np.linalg.LinAlgError:
        print("❌ Sn is singular, cannot invert.")
        return None
    Sn_inv_6_6 = np.zeros((6, 6)) 
    
    Sn_inv_6_6[0, 0] = Sn_inv[0, 0]
    Sn_inv_6_6[0, 3] = Sn_inv[0, 1]
    Sn_inv_6_6[0, 4] = Sn_inv[0, 2]
    Sn_inv_6_6[0, 5] = Sn_inv[0, 3]
    
    Sn_inv_6_6[3, 0] = Sn_inv[1, 0]
    Sn_inv_6_6[3, 3] = Sn_inv[1, 1]
    Sn_inv_6_6[3, 4] = Sn_inv[1, 2]
    Sn_inv_6_6[3, 5] = Sn_inv[1, 3]

    Sn_inv_6_6[4, 0] = Sn_inv[2, 0]
    Sn_inv_6_6[4, 3] = Sn_inv[2, 1]
    Sn_inv_6_6[4, 4] = Sn_inv[2, 2]
    Sn_inv_6_6[4, 5] = Sn_inv[2, 3]

    Sn_inv_6_6[5, 0] = Sn_inv[3, 0]
    Sn_inv_6_6[5, 3] = Sn_inv[3, 1]
    Sn_inv_6_6[5, 4] = Sn_inv[3, 2]
    Sn_inv_6_6[5, 5] = Sn_inv[3, 3]


    return Sn_inv_6_6

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
    members = last_step['members']

    load_pt = {}
    for idx, pt in enumerate(points, start=1):
        load_pt[idx] = pt['force_moment']
    load_mem = defaultdict(list)
    for idx, mem in enumerate(members, start=1):
        load_mem[idx].append(mem[0]['force_moment'])
        load_mem[idx].append(mem[-1]['force_moment'])
    
    return load_pt,load_mem

def give_k(load_mem, K_current, members):
    K_next = copy.deepcopy(K_current)
    kp_loads = defaultdict(list)
    for member in members:
        memb_no = member['memb_no']
        kp1 = member['kp1']
        kp2 = member['kp2']
        mate1 = member['mate1']
        mate2 = member['mate2']  
        
        load_kp1 = np.array(load_mem[memb_no][0])
        load_kp2 = np.array(load_mem[memb_no][1])
        kp_loads[kp1].append(load_kp1)
        kp_loads[kp2].append(load_kp2)
    for kp,loads in kp_loads.items():
        kp_loads[kp] = sum(loads)/len(loads)
    # print(kp_loads)
    for kp,avg_load in kp_loads.items():
        material_id = kp

        # Find section matrix K for this material:
        section_matrix = None
        for mat in K_next:
            if int(mat['material_no']) == material_id:
                section_matrix = np.array(mat['matrix'])
        if section_matrix is None:
            raise ValueError(f"Material {mate1} not found in K_current!")
        # Compute gamma vector:
        gamma_vector = section_matrix @ avg_load

        # Compute Sn:
        Sn_inv = compute_Sn( gamma_vector[0], gamma_vector[3], gamma_vector[4], gamma_vector[5])
        if Sn_inv is not None:
            # Update this material's section matrix:
            for mat in K_next:
                if int(mat['material_no']) == material_id:
                    mat['matrix'] = Sn_inv.tolist()
            
    #     else:
    #         print(f"⚠️ Skipped update for Material {mate1} due to singular Sn.")
    # print(f"✅ Updated K matrix for Materials")
    return K_next


def save_k_matrices(K_current, itr, output_dir):
    k_matrix_dir = os.path.join(output_dir, "K_matrices")
    os.makedirs(k_matrix_dir, exist_ok=True)

    for mat in K_current:
        material_no = mat['material_no']
        material_dir = os.path.join(k_matrix_dir, f"material{material_no}")
        os.makedirs(material_dir, exist_ok=True)
        
        k_matrix_path = os.path.join(material_dir, f"K_iter{itr}.txt")
        np.savetxt(k_matrix_path, np.array(mat['matrix']), fmt='%.6e')
    print(f"✅ Saved K matrix for Iteration {itr}")

def extract_disp_load(output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic=False):
    """
    Extract displacement and force vectors for each point at the last timestep.

    Parameters:
    - output_path: path to the GEBT output file.
    - nstep: number of time steps (we use the last one).
    - n_kp: number of key points.
    - n_member: number of members.
    - ndivs_per_member: list of divisions per member.
    - is_dynamic: whether the analysis is dynamic.

    Returns:
    - disp_vector: {point_no: [ux, uy, uz, θx, θy, θz]}
    - load_vector: {point_no: [Fx, Fy, Fz, Mx, My, Mz]}
    """
    step_data = extract_data.parse_all_steps_from_firststyle(
        output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic
    )

    last_step = step_data[nstep]
    points = last_step['points']

    disp_vector = {}
    load_vector = {}
    for i,pt in enumerate(points):
        disp_vector[i] = np.array(pt['displacement'])  # [ux, uy, uz, θx, θy, θz]
        load_vector[i] = np.array(pt['force_moment'])  # [Fx, Fy, Fz, Mx, My, Mz]

    return disp_vector, load_vector


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
    save_k_matrices(K_current, 0, output_dir)

    # Total force & scaling per iteration
    total_force, force_per_iteration = extract_and_compute_forces(point_conditions, num_iterations)

    # Convergence variables
    disp_prev = None
    force_prev = None
    threshold = 1e-5
    residual_dir = os.path.join(output_dir, "residuals")
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
        disp,force= extract_disp_load(output_to_generate, nstep, n_kp, n_member, ndivs_per_member, is_dynamic)

        # Extract updated loads
        load,load_mem = extract_loads(output_to_generate, nstep, n_kp, n_member, ndivs_per_member, is_dynamic) 
        # Update stiffness matrix K
        
        K_next = give_k(load_mem, K_current, members)
        

        # Save current K matrix
        save_k_matrices(K_next, itr, output_dir)
        
        if K_next and K_current:
            convergence = True
            for i in range(len(K_next)):
                
                K_next_np = np.array(K_next[i]['matrix'])
                K_current_np = np.array(K_current[i]['matrix'])
            
             

                residual = np.abs(K_next_np - K_current_np)
                # print(residual)
                if np.any(residual > threshold):
                    convergence = False
                

                    break
            if convergence:
                print("K values converged")
                break 
        # print('k_next: ',K_next)
        # print('k_curr: ',K_current)
        K_current = K_next

            


    # Generate final input file with total original force and last K
    final_input = os.path.join(output_dir, f"{base_name}_final.dat")
    modify_input.modify_input_file_forces(input_file, final_input, num_iterations, force_per_iteration)
    modify_input.modify_section_matrix(final_input, final_input, K_current)
    print("\n")
    print("🏁 Final input file generated:", final_input)

    try:
        produce_output.run_gebt_case(input_file=final_input)
        print("✅ Final GEBT output generated.")
    except Exception as e:
        print(f"❌ Error running final GEBT case: {e}")

if __name__ == "__main__":
    input_file = "trial.dat"
    num_iterations = 10
    iteration_loop(input_file, num_iterations)
