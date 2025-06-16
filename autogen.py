def write_gebr_input(filename):
    with open(filename, 'w') as f:
        # Analysis control
        f.write("0 1 1\n")  # Static, 1 iteration, 1 step
        f.write("\n")  # Space after Analysis Control

        # Simulation range (required if time function is present)
        f.write("\n")  # Blank as it's static — but necessary

        # Mesh description
        nkp = 3
        nmemb = 2
        ncond_pt = 2  # Clamped + Force-applied
        nmate = 1
        nframe = 0
        ncond_mb = 0
        ndistr = 0
        ntimefun = 1
        ncurv = 0
        f.write(f"{nkp} {nmemb} {ncond_pt} {nmate} {nframe} {ncond_mb} {ndistr} {ntimefun} {ncurv}\n")
        f.write("\n")  # Space after Mesh Description

        # Keypoints
        keypoints = [
            (1, 0.0, 0.0, 0.0),
            (2, 0.5, 0.0, 0.0),
            (3, 1.0, 0.0, 0.0)
        ]
        for kp in keypoints:
            f.write(f"{kp[0]} {kp[1]} {kp[2]} {kp[3]}\n")
        f.write("\n")  # Space after Keypoints

        # Members
        members = [
            (1, 1, 2, 1, 1, 0, 1, 0),
            (2, 1, 2, 1, 1, 0, 1, 0),
            
        ]
        for memb in members:
            f.write(f"{memb[0]} {memb[1]} {memb[2]} {memb[3]} {memb[4]} {memb[5]} {memb[6]} {memb[7]}\n")
        f.write("\n")  # Space after Members

        # Point Conditions
        # 1st Point Condition (Clamped at KP1)
        f.write("1\n")  # KP No.
        f.write("1 2 3 4 5 6\n")  # All DOFs constrained
        f.write("0 0 0 0 0 0\n")  # Zero values
        f.write("0 0 0 0 0 0\n")  # No time function
        f.write("0 0 0 0 0 0\n")  # No follower
        f.write("\n")  # Space after 1st Point Condition

        # 2nd Point Condition (Force applied at KP3)
        f.write("3\n")  # KP No.
        f.write("7 8 9 10 11 12\n")  # Load DOFs
        f.write("0 100000 100000 0 0 0\n")  # Forces in Y,Z
        f.write("1 1 1 1 1 1\n")  # Time function 1 for all
        f.write("0 0 0 0 0 0\n")  # Follower load
        f.write("\n")  # Space after 2nd Point Condition

        # Material (Flexibility and Mass Matrix placeholders)
        f.write("1\n")  # Material number

        # Flexibility matrix (from your provided values)
        flexibility_matrix = [
            [1.9230769231E-08, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2.0689655172E-05, 0, 0],
            [0, 0, 0, 0, 5.7692307692E-06, 0],
            [0, 0, 0, 0, 0, 2.3076923077E-05]
        ]
        
        # Write Flexibility Matrix in GEBT format (row by row)
        for row in flexibility_matrix:
            f.write(" ".join([f"{val:.10E}" for val in row]) + "\n")

        f.write("\n")  # Space after Flexibility Matrix

        # Time Functions
        f.write("0 1\n")  # Simulation range 0 to 1
        f.write("1\n")  # Time function no 1
        f.write("0\n")  # Type 0 = piecewise linear
        f.write("0 1\n")  # Function defined in 0 to 1
        f.write("2\n")  # 2 points
        f.write("0 0\n")  # At t=0, value=0
        f.write("1 1\n")  # At t=1, value=1
        f.write("\n")  # Space after Time Function

# Generate the input file
write_gebr_input("gebr_auto_input.dat")
print("GEBT input file 'gebr_auto_input.dat' with proper spacing created successfully.")
