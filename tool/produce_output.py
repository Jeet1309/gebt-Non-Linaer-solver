import subprocess
from pathlib import Path
import os

def run_gebt_case(
    input_file,                 # Full path to the input .dat file
    gebt_exec="/home/jeetgurbani/gebt/build/gebt",  # Path to GEBT executable
    output_file=None            # Full path to save the output .dat.out file
):
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❗ Input file not found: {input_file}")
        return

    if output_file is None:
        # Default: output in the same dir with .dat.out extension
        output_file = input_path.with_suffix(".dat.out")
    else:
        output_file = Path(output_file)

    print(f"▶ Running GEBT on: {input_path.name}")

    # Run GEBT
    result = subprocess.run([gebt_exec, str(input_path)], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error running {input_path.name}:")
        print(result.stderr)
        return

    # GEBT likely produces 'input_file.dat.out' in the same dir as 'input_file'
    generated_out_file = input_path.with_suffix(".dat.out")

    if generated_out_file.exists():
        # Move to target output_file location
        os.rename(generated_out_file, output_file)
        print(f"✅ Output saved to: {output_file}")
    else:
        print(f"⚠️ Output not found for {input_path.name}")


# Entry point
if __name__ == "__main__":
    run_gebt_case("/home/jeetgurbani/gebt/test1.dat")
    
