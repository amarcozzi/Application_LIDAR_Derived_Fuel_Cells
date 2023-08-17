import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define global directories
WORKFLOW_PATH = os.getenv("BURN_CHAMBER_WORKFLOW_PATH")
if not WORKFLOW_PATH:
    WORKFLOW_PATH = Path("/home/anthony/fire/projects/tls_to_fds/workflow")
else:
    WORKFLOW_PATH = Path(WORKFLOW_PATH)
DAKOTA_INPUTS_PATH = Path(WORKFLOW_PATH, "dakota")
PYTHON_SCRIPTS_PATH = Path(WORKFLOW_PATH, "python-scripts")
TEMPLATES_PATH = Path(WORKFLOW_PATH, "templates")
DATA_PATH = os.getenv("BURN_CHAMBER_DATA_PATH")
if not DATA_PATH:
    DATA_PATH = Path("/mnt/Data/burn-chamber/data")
else:
    DATA_PATH = Path(DATA_PATH)
POINT_CLOUD_DATA_PATH = Path(DATA_PATH, "point_clouds")
VOXEL_DATA_PATH = Path(DATA_PATH, "voxels")
MLR_DATA_PATH = Path(DATA_PATH, "massloss")
MLR_OBSERVED_DATA_PATH = Path(MLR_DATA_PATH, "observed")
MLR_OUT_PATH = Path(MLR_DATA_PATH, "processed")


def read_parameters_file(fpath):
    """
    Takes in a DAKOTA parameters file and returns the parameter values in a dictionary
    """
    with open(fpath) as f:
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


def get_tree_name_from_eval_id(id):
    tree_number = int(id)
    with open(Path(VOXEL_DATA_PATH, "tree_list.json")) as fin:
        tree_list = json.load(fin)
    tree_name = tree_list[tree_number]
    return tree_name


def get_mlr_filename_from_tree_name(tname, mlr_files):
    for file_name in mlr_files:
        if tname == file_name.stem.split("_")[-1]:
            return file_name


def get_tree_mlr_data(fname):
    # Open the mass loss rate file
    with open(Path(MLR_OBSERVED_DATA_PATH, fname)) as fin:
        return pd.read_csv(fin, delimiter=",")


def get_observed_mass_loss(name):
    file_list = MLR_OUT_PATH.glob("*")
    mlr_filename = get_mlr_filename_from_tree_name(name, file_list)
    return get_tree_mlr_data(mlr_filename)


def get_simulated_mass_loss(id):
    with open(Path(DAKOTA_INPUTS_PATH, f"out-{id}_devc.csv")) as fin:
        in_data = pd.read_csv(fin, header=1)
    return in_data


def get_consumption(data):
    out_data = pd.DataFrame.copy(data)

    initial_mass = out_data.iloc[1]["foliage"]
    min_index = (out_data["Time"] - 30).abs().argmin()
    end_mass = out_data.iloc[min_index]["foliage"]
    return 1 - (end_mass / initial_mass)


def process_simulated_mass_loss(data):
    out_data = pd.DataFrame.copy(data)

    # Convert kg to g
    out_data["foliage"] *= 1000

    # Focus data on 0 to 30 second range
    out_data = out_data.loc[out_data["Time"] >= 0]
    out_data = out_data.loc[out_data["Time"] <= 30]

    # Compute cumulative difference
    out_data["diff"] = out_data.foliage.diff().fillna(0)
    out_data["diff"].where(out_data["diff"] < 0, 0, inplace=True)
    out_data["cum-diff"] = out_data["diff"].cumsum()

    return out_data


def compute_rmse(a, b):
    diff = np.square(a - b)
    sum = np.sum(diff)
    mean = sum / a.size
    return mean


def compute_mae(a, b):
    return np.abs(a - b)


def compare_curves(o_df, s_df, name, eval_num):
    # Turn the relevant series into numpy arrays
    observed_time = o_df["time"].to_numpy()
    observed_data = o_df["cum-diff"].to_numpy()
    simulated_time = s_df["Time"].to_numpy()
    simulated_data = s_df["cum-diff"].to_numpy()

    # Create a 1D array to hold the desired times at which to interpolate the two curves
    desired_times = np.linspace(0, 30, 30)

    # Interpolate the observed data to get values for the desired times
    observed_interpolated = np.interp(desired_times, observed_time, observed_data)

    # Interpolate the simulated data to get values for the desired times
    simulated_interpolated = np.interp(desired_times, simulated_time, simulated_data)

    # Compute loss metrics
    rmse = compute_rmse(observed_interpolated, simulated_interpolated)
    mae = compute_mae(observed_interpolated, simulated_interpolated)[-1]

    return rmse, mae


def compute_loss(p, eval_num):
    tree_name = get_tree_name_from_eval_id(p["tree"])
    observed_data = get_observed_mass_loss(tree_name)
    simulated_data = get_simulated_mass_loss(eval_num)
    consumption = get_consumption(simulated_data)
    simulated_data = process_simulated_mass_loss(simulated_data)
    rmse, mae = compare_curves(observed_data, simulated_data, tree_name, eval_num)
    return rmse, mae, consumption


def write_loss_to_results_file(rmse, mae, consumption, eval_num):
    with open(Path(DAKOTA_INPUTS_PATH, f"results.tmp.{eval_num}"), "w") as fout:
        fout.write(str(rmse))
        fout.write("\n")
        fout.write(str(mae))
        fout.write("\n")
        fout.write(str(consumption))


def main(args):
    if len(args) > 1:
        params_fname = args[1]
        eval_num = args[2]
    else:
        params_fname = Path(PYTHON_SCRIPTS_PATH, "test_params")
        eval_num = "1"

    params = read_parameters_file(params_fname)
    rmse, mae, consumption = compute_loss(params, eval_num)
    write_loss_to_results_file(rmse, mae, consumption, eval_num)


if __name__ == "__main__":
    """
    Computes a loss function between the observed and simulated mass loss curves.

    System Arguments:
    0 - python file name
    1 - name of the parameters file
    2 - evaluation number
    """
    main(sys.argv)
