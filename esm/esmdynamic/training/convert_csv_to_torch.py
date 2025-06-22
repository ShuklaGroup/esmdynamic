'''Author: Diego E. Kleiman (Shukla Group, UIUC)

Use this script to convert .csv files extracted from `rcsb/rcsb.tar.gz` or `mdcath/mdcath.tar.gz` into torch.Tensor (.pt) format required for training.

Required packages: numpy, torch, tqdm.
'''

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def csv_to_tensor(csv_path: Path) -> torch.Tensor:
    """
    Load a 1D CSV file with one integer per line and convert it to a PyTorch tensor.

    Args:
        csv_path (Path): Path to the CSV file.

    Returns:
        torch.Tensor: 1D tensor of dtype torch.int
    """
    array = np.loadtxt(csv_path, dtype=np.int32)
    array = np.ravel(array)
    return torch.tensor(array, dtype=torch.int)

def convert_all_csvs(root_dir: Path):
    """
    Convert all dynamic_contacts.csv files under subdirectories of root_dir
    into .pt torch tensor files.

    Args:
        root_dir (Path): The root directory to search in.
    """
    csv_files = list(root_dir.glob("**/dynamic_contacts.csv"))

    for csv_path in tqdm(csv_files, desc="Converting CSVs"):
        tensor = csv_to_tensor(csv_path)
        output_path = csv_path.with_suffix(".pt")
        torch.save(tensor, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dynamic_contacts.csv files to PyTorch .pt format.")
    parser.add_argument("root", type=str, help="Path to the root folder containing subfolders with CSV files.")
    args = parser.parse_args()

    root_dir = Path(args.root)
    if not root_dir.is_dir():
        raise ValueError(f"{args.root} is not a valid directory")

    convert_all_csvs(root_dir)