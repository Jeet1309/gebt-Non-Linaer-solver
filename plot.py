import extract_data
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_force_and_moment(ax, x, y, force_vec, moment_vec, scale=0.1):
    """
    Draw force and moment arrows on the plot.
    - force_vec: [Fx, Fy, Fz] 
    - moment_vec: [Mx, My, Mz]
    Only Fx, Fy, Fz shown in 2D view.
    """
    Fx, Fy, Fz = force_vec[:3]
    Mx, My, Mz = moment_vec[:3]

    # --- Plot Force (as straight arrow) ---
    if abs(Fx) > 1e-6 or abs(Fy) > 1e-6:  # significant force
        ax.arrow(x, y, Fx*scale, Fy*scale, 
                 head_width=0.02, head_length=0.04, fc='g', ec='g')
        ax.text(x + Fx*scale*1.1, y + Fy*scale*1.1, 
                f"{np.linalg.norm([Fx,Fy]):.1f}N", color='g', fontsize=8)

    # --- Plot Moment (as circular arrow symbol) ---
    moment_mag = np.linalg.norm([Mx, My, Mz])
    if moment_mag > 1e-6:
        # Draw a symbolic circle to represent moment (purely visual)
        circle = plt.Circle((x, y), 0.05, color='m', fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.text(x + 0.06, y + 0.06, f"{moment_mag:.1f}Nm", color='m', fontsize=8)

def interactive_beam_animation(steps,file, eigen_data=None, scale=1.0, is_eigen=False):
    """
    Interactive animation for both normal steps and eigenmodes.
    Press 'n' to go to the next frame (either step or eigenmode).
    
    Parameters:
    - steps: parsed steady/dynamic step data
    - eigen_data: parsed eigenmode data (if any)
    - scale: scaling factor
    - is_eigen: bool, if True then eigen_data will also be plotted
    """
    # Total frames = steps + eigenmodes
    total_steps = len(steps) # because steps start at 1
    total_modes = len(eigen_data) if (is_eigen and eigen_data is not None) else 0
    total_frames = total_steps + total_modes

    current_frame = [1]  # mutable for the callback

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 2 subplots: XZ and XY

    def plot_step_or_eigen(frame):
        ax[0].clear()
        ax[1].clear()

        if frame <= total_steps:
            # Plot Step data (steady/dynamic)
            step = steps[frame]
            members = step["members"]
            points = step["points"]
            title = f"{file} - Step {frame}/{total_steps}"
        else:
            # Plot Eigenmode
            eigen_mode_num = frame - total_steps
            eigen_mode = eigen_data[eigen_mode_num]
            members = eigen_mode["members"]
            points = eigen_mode["points"]
            title = f"{file} - Eigenmode {eigen_mode_num}/{total_modes}"

        # Member segments
        undeformed_xz = []
        deformed_xz = []
        undeformed_xy = []
        deformed_xy = []
        
        for member in members:
            for seg in member:
                x, y, z = seg["position"]
                dx, dy, dz = seg["displacement"][:3]
                undeformed_xz.append((x, z))
                deformed_xz.append((x + dx * scale, z + dz * scale))
                undeformed_xy.append((x, y))
                deformed_xy.append((x + dx * scale, y + dy * scale))

        # Keypoints
        point_und_xz = []
        point_def_xz = []
        point_und_xy = []
        point_def_xy = []

        for pt in points:
            x, y, z = pt["position"]
            dx, dy, dz = pt["displacement"][:3]
            point_und_xz.append((x, z))
            point_def_xz.append((x + dx * scale, z + dz * scale))
            point_und_xy.append((x, y))
            point_def_xy.append((x + dx * scale, y + dy * scale))

        # Unpack for plotting
        x_und_xz, z_und = zip(*undeformed_xz)
        x_def_xz, z_def = zip(*deformed_xz)
        x_und_xy, y_und = zip(*undeformed_xy)
        x_def_xy, y_def = zip(*deformed_xy)
        px_und_xz, pz_und = zip(*point_und_xz)
        px_def_xz, pz_def = zip(*point_def_xz)
        px_und_xy, py_und = zip(*point_und_xy)
        px_def_xy, py_def = zip(*point_def_xy)

        # Plot XZ view (Top view)
        ax[0].plot(x_und_xz, z_und, 'k--', linewidth=1, label="Undeformed")
        ax[0].plot(x_def_xz, z_def, 'r-', linewidth=2, label=f"Deformed (x{scale:.0e})")
        ax[0].plot(px_und_xz, pz_und, 'ko', label="Keypoints (Undeformed)")
        ax[0].plot(px_def_xz, pz_def, 'ro', label="Keypoints (Deformed)")
        ax[0].set_xlabel("X Position")
        ax[0].set_ylabel("Z Position")
        ax[0].set_title("XZ View (Top View)")
        ax[0].legend()
        ax[0].grid(True)
        ax[0].axis("equal")

        # Plot XY view (Side view)
        ax[1].plot(x_und_xy, y_und, 'k--', linewidth=1, label="Undeformed")
        ax[1].plot(x_def_xy, y_def, 'b-', linewidth=2, label=f"Deformed (x{scale:.0e})")
        ax[1].plot(px_und_xy, py_und, 'ko', label="Keypoints (Undeformed)")
        ax[1].plot(px_def_xy, py_def, 'bo', label="Keypoints (Deformed)")
        ax[1].set_xlabel("X Position")
        ax[1].set_ylabel("Y Position")
        ax[1].set_title("XY View (Side View)")
        ax[1].legend()
        ax[1].grid(True)
        ax[1].axis("equal")
        
        fig.suptitle(title)
        fig.tight_layout()
        fig.canvas.draw()
    def on_key(event):
        if event.key == 'n':
            if current_frame[0] < total_frames:
                current_frame[0] += 1
                plot_step_or_eigen(current_frame[0])
            else:
                print("✅ End of all steps and eigenmodes.")
                plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plot_step_or_eigen(current_frame[0])
    plt.show()

if __name__ == "__main__":
    
    input_files = [f"trial_output_iteration/trial_input_iteration/trial_iter{i}.dat" for i in range(1,11)]  
    output_files = [f"trial_output_iteration/trial_gebt_outputs/trial_iter{i}.dat.out" for i in range(1,11)]

    for i in range(len(input_files)):
        print(f"Processing: {input_files[i]}")
        try:
            metadata, members,point_conditions,_ = extract_data.process_input_file(input_files[i],is_space=True)
        except:
            metadata, members,point_conditions,_ = extract_data.process_input_file(input_files[i],is_space=False)
        n_kp = metadata["n_kp"]
        n_member = metadata["n_member"]
        nstep = metadata["nstep"]
        ndivs_per_member = [m["ndiv"] for m in members]
        
        is_dynamic = (metadata["analysis_flag"] != 0)
        is_eigen = (metadata["analysis_flag"] == 3)
        basename=os.path.splitext(os.path.basename(input_files[i]))[0]
        if is_eigen:
            nev = metadata["n_ev"]
            steps, eigen_data = extract_data.parse_all_steps_from_firststyle_for_eigen(
                output_files[i], nstep, n_kp, n_member, ndivs_per_member, nev
            )
            
            interactive_beam_animation(steps,basename, eigen_data, scale=1000, is_eigen=True)

        else:
            steps = extract_data.parse_all_steps_from_firststyle(
                output_files[i], nstep, n_kp, n_member, ndivs_per_member, is_dynamic
            )
            interactive_beam_animation(steps,basename, scale=1, is_eigen=False)
