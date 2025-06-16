import re

def process_input_file(input_path,is_space):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    i = 0
    analysis_line = lines[i].strip().split()
    analysis_flag = int(analysis_line[0])
    niter = int(analysis_line[1])
    nstep = int(analysis_line[2])
    i += 1

    nev = None
    if analysis_flag != 0:
        i += 4
        if analysis_flag == 3:
            # i+=1
            nev = int(lines[i].strip())
            i += 1
    i += 1  # move to object count line
    object_line = lines[i].strip().split()
    n_kp = int(object_line[0])
    n_member = int(object_line[1])
    n_cond_pt = int(object_line[2])
    n_mate = int(object_line[3])
    n_frame = int(object_line[4])
    n_memb_load = int(object_line[5])
    n_distributed_load = int(object_line[6])
    n_time_fun = int(object_line[7])
    n_curve = int(object_line[8])
    
    
    
    i += 1
    i += n_kp# skip keypoints
    if(is_space):
        i+=1 # next line 
    # Skip 2 blank lines
    print(1,lines[i])
    print(2,lines[i+1])
    if lines: i+=1  #flag ==0
    i+=1
    members = []
    print(3,lines[i])
    print(4,lines[i+1])
    for _ in range(n_member):
        parts = lines[i].strip().split()
        member = {
            "memb_no": int(parts[0]),
            "kp1": int(parts[1]),
            "kp2": int(parts[2]),
            "mate1": int(parts[3]),
            "mate2": int(parts[4]),
            "frame": int(parts[5]),
            "ndiv": int(parts[6]),
            "curv": int(parts[7])
        }
        members.append(member)
        i += 1

    metadata = {
        "analysis_flag": analysis_flag,
        "description": {
            0: "Static Analysis",
            1: "Steady-State",
            2: "Transient Dynamic",
            3: "Eigenvalue"
        }.get(analysis_flag, "Unknown"),
        "niter": niter,
        "linearity": "Linear" if niter == 1 else "Nonlinear",
        "nstep": nstep,
        "n_kp": n_kp,
        "n_member": n_member,
        "n_cond_pt":n_cond_pt,
        "n_mate":n_mate,
    }
    if nev is not None:
        metadata["n_ev"] = nev
    i+=1


    # for()
    # print(lines[i])
    point_conditions = []

    for _ in range(n_cond_pt):
        point_no = int(lines[i][0])
        dof = list(map(int, lines[i+1].strip().split()))
        value = list(map(float, lines[i+2].strip().split()))
        time_function = list(map(int, lines[i+3].strip().split()))
        follower = list(map(int, lines[i+4].strip().split()))
        point_conditions.append({
            'point': point_no,
            'dof': dof,
            'value': value,
            'time_function': time_function,
            'follower': follower
        })
        i += 6  # move to next condition block or section matrix start
    print(f"Reading at line {i}: '{repr(lines[i])}'")
    # Parse Section Matrices (for n_material)
    section_matrices = []
    for _ in range(n_mate):
        material_no = int(lines[i][0])
        i += 1
        matrix = []
        for _ in range(6):
            row = list(map(float, lines[i].strip().split()))
            matrix.append(row)
            i += 1
        section_matrices.append({
            'material_no': material_no,
            'matrix': matrix
        })

    return metadata, members, point_conditions, section_matrices

def parse_all_steps_from_firststyle(output_path, nstep, n_kp, n_member, ndivs_per_member, is_dynamic=False):
    with open(output_path, 'r') as f:
        lines = f.readlines()

    step_data = {}
    i = 4  # Start from line 5 (0-indexed)

    for step in range(1, nstep + 1):
        # Skip "Step #" line if present
        if lines[i].strip().startswith("Step #"):
            i += 1

        step_points = []
        step_members = []

        # --- Parse n_kp points ---
        for _ in range(n_kp):
            while not lines[i].strip().startswith("Point #:"):
                i += 1
            i += 2  # skip dashed line

            pos = list(map(float, lines[i].strip().split()))
            disp = list(map(float, lines[i + 1].strip().split()))
            force = list(map(float, lines[i + 2].strip().split()))
            i += 3

            step_points.append({
                "position": pos,
                "displacement": disp,
                "force_moment": force
            })

            if lines[i].strip() == "":
                i += 1

        # --- Parse n_member sets of segments ---
        for m in range(n_member):
            if lines[i].strip().startswith("Member #:"):
                i += 2  # skip dashed line
                member_segments = []

                for _ in range(ndivs_per_member[m]):
                    pos = list(map(float, lines[i].strip().split()))
                    disp = list(map(float, lines[i + 1].strip().split()))
                    force = list(map(float, lines[i + 2].strip().split()))
                    i += 3

                    momenta = None
                    if is_dynamic:
                        momenta = list(map(float, lines[i].strip().split()))
                        i += 1

                    member_segments.append({
                        "position": pos,
                        "displacement": disp,
                        "force_moment": force,
                        "momenta": momenta
                    })

                    if i < len(lines) and lines[i].strip() == "":
                        i += 1

                step_members.append(member_segments)

        # Store step
        step_data[step] = {
            "points": step_points,
            "members": step_members
        }

        # Skip extra blank lines (if any) before next step
        while i < len(lines) and lines[i].strip() == "":
            i += 1

    return step_data
def parse_all_steps_from_firststyle_for_eigen(output_path, nstep, n_kp, n_member, ndivs_per_member,nev, is_eigen=False):
    with open(output_path, 'r') as f:
        lines = f.readlines()

    step_data_stedy = {}
    i = 4  # Start from line 5 (0-indexed)

    for step in range(1, nstep + 1):
        # Skip "Step #" line if present
        if lines[i].strip().startswith("Step #"):
            i += 1

        step_points = []
        step_members = []

        # --- Parse n_kp points ---
        for _ in range(n_kp):
            while not lines[i].strip().startswith("Point #:"):
                i += 1
            i += 2  # skip dashed line

            pos = list(map(float, lines[i].strip().split()))
            disp = list(map(float, lines[i + 1].strip().split()))
            force = list(map(float, lines[i + 2].strip().split()))
            i += 3

            step_points.append({
                "position": pos,
                "displacement": disp,
                "force_moment": force
            })

            if lines[i].strip() == "":
                i += 1

        # --- Parse n_member sets of segments ---
        for m in range(n_member):
            if lines[i].strip().startswith("Member #:"):
                i += 2  # skip dashed line
                member_segments = []

                for _ in range(ndivs_per_member[m]):
                    pos = list(map(float, lines[i].strip().split()))
                    disp = list(map(float, lines[i + 1].strip().split()))
                    force = list(map(float, lines[i + 2].strip().split()))
                    momenta = list(map(float, lines[i + 3].strip().split()))
                    i += 4

                    member_segments.append({
                        "position": pos,
                        "displacement": disp,
                        "force_moment": force,
                        "momenta": momenta
                    })

                    if i < len(lines) and lines[i].strip() == "":
                        i += 1

                step_members.append(member_segments)

        # Store step
        step_data_stedy[step] = {
            "points": step_points,
            "members": step_members
        }

        # Skip extra blank lines (if any) before next step
        while i < len(lines) and lines[i].strip() == "":
            i += 1
    

    eigen_data ={}
    for eigen_value in range(1,nev+1):
        if lines[i].strip().startswith("Eigenvalue #"):
            i += 1
        eigen_points = []
        eigen_members = []
        eigen_vector  = list(map(float, lines[i].strip().split()))
        

        # --- Parse n_kp points ---
        for _ in range(n_kp):
            while not lines[i].strip().startswith("Point #:"):
                i += 1
            i += 2  # skip dashed line

            pos = list(map(float, lines[i].strip().split()))
            disp = list(map(float, lines[i + 1].strip().split()))
            force = list(map(float, lines[i + 2].strip().split()))
            i += 3

            eigen_points.append({
                "position": pos,
                "displacement": disp,
                "force_moment": force
            })

            if lines[i].strip() == "":
                i += 1

        # --- Parse n_member sets of segments ---
        for m in range(n_member):
            if lines[i].strip().startswith("Member #:"):
                i += 2  # skip dashed line
                member_segments = []

                for _ in range(ndivs_per_member[m]):
                    pos = list(map(float, lines[i].strip().split()))
                    disp = list(map(float, lines[i + 1].strip().split()))
                    force = list(map(float, lines[i + 2].strip().split()))
                    momenta = list(map(float, lines[i + 3].strip().split()))
                    i += 4

                    member_segments.append({
                        "position": pos,
                        "displacement": disp,
                        "force_moment": force,
                        "momenta": momenta
                    })

                    if i < len(lines) and lines[i].strip() == "":
                        i += 1

                eigen_members.append(member_segments)

        # Store step
        
        eigen_data[eigen_value] ={
            "Eigenvector" : eigen_vector,
            "points": eigen_points,
            "members": eigen_members
            
        }

        # Skip extra blank lines (if any) before next step
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        
        # pass


    return step_data_stedy,eigen_data
def print_all_gebt_output(step_data):
    for step_num, step_content in step_data.items():
        print(f"\n📦 STEP {step_num}")
        print("🔹 Points:")
        for idx, pt in enumerate(step_content["points"], 1):
            print(f"  Point {idx}:")
            print(f"    Position      : {pt['position']}")
            print(f"    Displacement  : {pt['displacement']}")
            print(f"    Force & Moment: {pt['force_moment']}")

        print("🔩 Members:")
        for m_id, member_segments in enumerate(step_content["members"], 1):
            print(f"  Member {m_id}:")
            for seg_id, seg in enumerate(member_segments, 1):
                print(f"    Segment {seg_id}:")
                print(f"      Position      : {seg['position']}")
                print(f"      Displacement  : {seg['displacement']}")
                print(f"      Force & Moment: {seg['force_moment']}")
                if seg['momenta'] is not None:
                    print(f"      Momenta       : {seg['momenta']}")
def print_all_gebt_output_eigen(step_data, eigen_data):
    # Print Steady/Transient Steps Data
    for step_num, step_content in step_data.items():
        print(f"\n📦 STEP {step_num}")
        print("🔹 Points:")
        for idx, pt in enumerate(step_content["points"], 1):
            print(f"  Point {idx}:")
            print(f"    Position      : {pt['position']}")
            print(f"    Displacement  : {pt['displacement']}")
            print(f"    Force & Moment: {pt['force_moment']}")

        print("🔩 Members:")
        for m_id, member_segments in enumerate(step_content["members"], 1):
            print(f"  Member {m_id}:")
            for seg_id, seg in enumerate(member_segments, 1):
                print(f"    Segment {seg_id}:")
                print(f"      Position      : {seg['position']}")
                print(f"      Displacement  : {seg['displacement']}")
                print(f"      Force & Moment: {seg['force_moment']}")
                if seg['momenta'] is not None:
                    print(f"      Momenta       : {seg['momenta']}")

    # Print Eigenvalue Data
    for eigen_num, eigen_content in eigen_data.items():
        print(f"\n🎵 Eigenvalue Mode {eigen_num}")
        print(f"Eigenvector: {eigen_content['Eigenvector']}")
        
        print("🔹 Points:")
        for idx, pt in enumerate(eigen_content["points"], 1):
            print(f"  Point {idx}:")
            print(f"    Position      : {pt['position']}")
            print(f"    Displacement  : {pt['displacement']}")
            print(f"    Force & Moment: {pt['force_moment']}")

        print("🔩 Members:")
        for m_id, member_segments in enumerate(eigen_content["members"], 1):
            print(f"  Member {m_id}:")
            for seg_id, seg in enumerate(member_segments, 1):
                print(f"    Segment {seg_id}:")
                print(f"      Position      : {seg['position']}")
                print(f"      Displacement  : {seg['displacement']}")
                print(f"      Force & Moment: {seg['force_moment']}")
                if seg['momenta'] is not None:
                    print(f"      Momenta       : {seg['momenta']}")

if __name__ == "__main__":
    input_files = ["gebr_auto_input.dat"]  # Example file
    output_files = ["gebr_auto_input.dat.out"]

    for i in range(len(input_files)):
        print(input_files[i])
        try:
            metadata, members, point_conditions,section_matrices= process_input_file(input_files[i], is_space=True)
        except:
            metadata, members, point_conditions,section_matrices = process_input_file(input_files[i], is_space=False)

        n_kp = metadata["n_kp"]
        n_member = metadata["n_member"]
        nstep = metadata["nstep"]
        ndivs_per_member = [m["ndiv"] for m in members]

        is_dynamic = (metadata["analysis_flag"] != 0)
        is_eigen = (metadata["analysis_flag"] == 3)

        print("🔍 Members:")
        print(members)
        print("\n📋 Metadata:")
        print(metadata)
        print("\n📌 Point Conditions:")
        print(point_conditions)
        print("\n🧩 Section Matrices:")
        print(section_matrices)

        # if(is_eigen):
        #     nev= metadata["n_ev"]
        #     # print(nev)
        #     steps,eigens = parse_all_steps_from_firststyle_for_eigen(
        #     output_files[i],
        #     nstep,
        #     n_kp,
        #     n_member,
        #     ndivs_per_member,
        #     nev,
        #     is_eigen
        #     )
        #     print_all_gebt_output_eigen(steps,eigens)
            
        # else:
        #     steps = parse_all_steps_from_firststyle(
        #         output_files[i],
        #         nstep,
        #         n_kp,
        #         n_member,
        #         ndivs_per_member,
        #         is_dynamic
        #     )
        #     print_all_gebt_output(steps)


        
        # print(steps.items())
