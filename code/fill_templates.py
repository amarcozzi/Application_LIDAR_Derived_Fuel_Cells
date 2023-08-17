import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from string import Template
from math import ceil, floor

# Define global directories
WORKFLOW_PATH = os.getenv("BURN_CHAMBER_WORKFLOW_PATH")
DAKOTA_INPUTS_PATH = Path(WORKFLOW_PATH, "dakota")
PYTHON_SCRIPTS_PATH = Path(WORKFLOW_PATH, "python-scripts")
TEMPLATES_PATH = Path(WORKFLOW_PATH, "templates")
DATA_PATH = os.getenv("BURN_CHAMBER_DATA_PATH")
DATA_PATH = Path(DATA_PATH)
POINT_CLOUD_DATA_PATH = Path(DATA_PATH, "point_clouds")
SEGMENTED = True
if SEGMENTED:
    VOXEL_DATA_PATH = Path(DATA_PATH, "voxels", "ref-001-segmented")
else:
    VOXEL_DATA_PATH = Path(DATA_PATH, "voxels", "ref-001")
VOXEL_SIZE = 0.01
SIMULATION_CELL_SIZE = 0.02


def main(args):
    if len(args) > 1:
        params_fname = args[1]
        eval_num = args[2]
        script_dir = args[3]
    else:
        params_fname = Path(PYTHON_SCRIPTS_PATH, "test_params")
        eval_num = "1"
        script_dir = Path(PYTHON_SCRIPTS_PATH)

    params = read_parameters_file(f"params.in.{eval_num}", script_dir)

    # Get a tree name from the tree number
    tree_number = int(params["tree"])
    with open(Path(VOXEL_DATA_PATH.parent, "tree_list.json")) as fin:
        tree_list = json.load(fin)
    tree_name = tree_list[tree_number]

    generate_fds_input_file(params, eval_num, tree_name)
    generate_submit_file(eval_num, tree_name)


def read_parameters_file(name, dir):
    """
    Takes in a DAKOTA parameters file and returns the parameter values in a dictionary
    """
    with open(f"{dir}/{name}") as f:
        lines = f.readlines()
    num_variables = int(lines[0].split(" ")[-2])
    params = {}
    for i in range(1, num_variables + 1):
        key, value = parse_parameters_line(lines[i])
        params[key] = value

    return params


def parse_parameters_line(line):
    split_line = line.split(" ")
    k = split_line[-1].strip()
    v = split_line[-2]
    return k, v


def get_predicted_weights(current_tree_name):
    # Determine if the tree is a spruce or pondo
    if "S" in current_tree_name:
        needle_filename = "pien_model_needle.pkl"
        tree_filename = "pien_model_tree.pkl"
    else:
        needle_filename = "pipo_model_needle.pkl"
        tree_filename = "pipo_model_tree.pkl"
    # Load the regression model
    needle_model = pickle.load(open(needle_filename, "rb"))
    stem_model = pickle.load(open(tree_filename, "rb"))

    # Load the lab data
    data_path = PYTHON_SCRIPTS_PATH / "notebooks" / "data"
    lab_df = pd.read_csv(data_path / "lab_data.csv")

    # Get the row where name matches TREE_ID
    series = lab_df.loc[lab_df["TREE_ID"] == current_tree_name]

    # Predict needle and stem weights
    x = series[["DIA"]]
    pred_needle_weight = needle_model.predict(x).values[0]
    pred_stem_weight = stem_model.predict(x).values[0]

    return pred_needle_weight, pred_stem_weight


def write_fuel_cells(resolution, mass, tree_name, component):
    # Determine which voxel data to load
    if component in ("stem", "veg"):
        coords_path = Path(
            VOXEL_DATA_PATH, tree_name, f"{component}_coords_{resolution}.npy"
        )
        counts_path = Path(
            VOXEL_DATA_PATH, tree_name, f"{component}_counts_{resolution}.npy"
        )

    else:
        coords_path = Path(VOXEL_DATA_PATH, tree_name, f"coords_{resolution}.npy")
        counts_path = Path(VOXEL_DATA_PATH, tree_name, f"counts_{resolution}.npy")

    # Determine which fuel cell component to write
    if component == "veg":
        part_id = "foliage"
    elif component == "stem":
        part_id = "large roundwood"
    else:
        part_id = "foliage"

    # Load the voxel coordinates and cell counts
    coords = np.load(str(coords_path))
    counts = np.load(str(counts_path))

    # Write homogenous cylinder
    if resolution == "0":
        # Get the dimensions of the cylinder
        x_radius = np.max(np.abs(coords[:, 0]))
        y_radius = np.max(np.abs(coords[:, 1]))
        radius = max(x_radius, y_radius)
        height = np.max(coords[:, 2])

        # Compute the volume of the cylinder
        cylinder_volume = np.pi * np.square(radius) * height

        # Compute the dry bulk density
        dry_bulk_density = mass / cylinder_volume

        # Write the cylinder fuel object to the FDS input file
        init_lines = [
            f"&INIT PART_ID='{part_id}', XYZ=0.0,0.0,0.05, RADIUS={radius}, HEIGHT={height}, SHAPE='CYLINDER', N_PARTICLES_PER_CELL=1, DRY=T, MASS_PER_VOLUME={dry_bulk_density} /"
        ]

    # Write LIDAR derived fuels
    else:
        # Compute the per voxel packing ratio
        resolution_int = int(resolution)
        voxel_size = VOXEL_SIZE * 2 ** (int(resolution_int))
        counts = np.array(counts, dtype=float)
        bulk_density = (counts * (mass / counts.sum())) / voxel_size**3

        # Write the fuel cell line for each coordinate
        init_lines = []
        for k in range(len(coords)):
            c0 = coords[k] - voxel_size / 2.0
            c1 = coords[k] + voxel_size / 2.0
            bd = bulk_density[k]
            init_lines.append(get_init_str(part_id, c0, c1, 1, bd))

    return init_lines


def generate_fds_input_file(params, eval_num, tree_name):
    # Store values in a dictionary for string formatting
    format_data = {}

    # Generate chid and title
    format_data["chid"] = f"out-{eval_num}"
    format_data["title"] = f"Evaluation {eval_num} with tree {tree_name}"

    # Add fuel moisture content to format dictionary
    format_data["fmc"] = "{:.4f}".format(float(params["fmc"]))

    # Determine if the tree receives the high or low burner treatment
    with open(Path(DATA_PATH, "tree_rx_map.json"), "r") as fin:
        rx_map = json.load(fin)
    burner_rx = rx_map[tree_name]["burn"]

    # Assign a HRRPUA to reach the desired HRR for a specified burner treatment. These numbers below are in
    # kW/m^2, and come from the compute_hrrpua_by_burner.py script in utils.
    if burner_rx == "Low":
        format_data["hrrpua"] = 516.2293005634149
    else:
        format_data["hrrpua"] = 1032.4586011268298

    resolution_factor_str = params["resolution"]
    needle_mass = float(params["mass"])
    _, stem_mass = get_predicted_weights(tree_name)
    stem_mass /= 1000.0  # Convert stem mass from g to kg
    if SEGMENTED:
        veg_lines = write_fuel_cells(
            resolution_factor_str, needle_mass, tree_name, "veg"
        )
        stem_lines = write_fuel_cells(
            resolution_factor_str, stem_mass, tree_name, "stem"
        )
        init_lines = veg_lines + stem_lines
    else:
        init_lines = write_fuel_cells(
            resolution_factor_str, needle_mass, tree_name, "all"
        )

    # Add the &INIT lines to the template data dictionary
    format_data["init_lines"] = "\n".join(init_lines)

    # Write the format data dictionary to the input template
    with open(Path(TEMPLATES_PATH, "input_template.fds"), "r") as ftemp:
        with open(
            Path(DAKOTA_INPUTS_PATH, tree_name, f"input-{eval_num}.fds"), "w"
        ) as fout:
            src = Template(ftemp.read())
            result = src.substitute(format_data)
            fout.write(result)


def get_init_str(name, c0, c1, n, dry_bulk_density):
    """
    Utility function for writing particles to the fds file
    """
    line = (
        "&INIT PART_ID='{:}', XB={:.4f},{:4f},{:.4f},{:.4f},{:.4f},{:.4f}, N_PARTICLES_PER_CELL={:}, "
        "DRY=T, MASS_PER_VOLUME={:.6f} / "
    )
    line = line.format(
        name, c0[0], c1[0], c0[1], c1[1], c0[2], c1[2], n, dry_bulk_density
    )
    return line


def generate_submit_file(eval_num, tree_name):
    """
    generates a submit.sh batch file
    """
    format_data = {
        "job_name": f"eval-{eval_num}",
        "output": str(Path(DAKOTA_INPUTS_PATH, tree_name, f"eval-{eval_num}.out")),
        "error": str(Path(DAKOTA_INPUTS_PATH, tree_name, f"eval-{eval_num}.err")),
        "partition": os.getenv("BURN_CHAMBER_PARTITION"),
        "sim_path": str(Path(DAKOTA_INPUTS_PATH, tree_name, f"input-{eval_num}.fds")),
    }

    # write the dictionary values to the template
    with open(Path(TEMPLATES_PATH, "submit_template.txt"), "r") as ftemp:
        with open(
            Path(DAKOTA_INPUTS_PATH, tree_name, f"submit-{eval_num}.sh"), "w"
        ) as fout:
            src = Template(ftemp.read())
            result = src.substitute(format_data)
            fout.write(result)


if __name__ == "__main__":
    main(sys.argv)
