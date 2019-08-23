import subprocess
import glob
import os

import pandas as pd


def load_ped_datasets() -> list[str]:
    """Returns paths after downloading pedestrian datasets."""
    if not os.path.exists("datasets"):
        subprocess.call(
            ["wget", "https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip"]
        )
        subprocess.call(["unzip", "-q", "datasets.zip"])
        subprocess.call(["rm", "-rf", "datasets.zip"])
    else:
        print("Directory 'datasets' exists, skipping download")

    return glob.glob(f"datasets/*/*")


def load_ped_data(dataset_name=None, aspaths=False) -> dict:
    """Returns pedestrian (ETH, Zara1, Zara2, Univ, Hotel) datasets as dataframe or as paths.

    Args:
        dataset_name: Optional(str) - returns specific dataset
                        eth
                        zara1
                        zara2
                        univ
                        hotel
        aspaths: (bool) - Returns paths only

    Returns:
        paths/dfs (dict) - train/val/test split for paths or dfs, depending on `aspaths` value


    Paths are .txt files with format <frame_id> <ped_id> <x> <y>.
    """
    paths = load_ped_datasets()

    if dataset_name:
        # Get subset of data
        paths = [path for path in paths if dataset_name in path]

    train_dir = [path for path in paths if "train" in path][0]
    val_dir = [path for path in paths if "val" in path][0]
    test_dir = [path for path in paths if "test" in path][0]

    train_paths = glob.glob(os.path.join(train_dir, "*.txt"))
    val_paths = glob.glob(os.path.join(val_dir, "*.txt"))
    test_paths = glob.glob(os.path.join(test_dir, "*.txt"))

    paths = {"train": train_paths, "val": val_paths, "test": test_paths}
    if aspaths:
        return paths

    col_names = ["frame_id", "ped_id", "x", "y"]
    dfs = {
        "train": [pd.read_csv(path, sep="\t", names=col_names) for path in train_paths],
        "val": [pd.read_csv(path, sep="\t", names=col_names) for path in train_paths],
        "test": [pd.read_csv(path, sep="\t", names=col_names) for path in train_paths],
    }
    return dfs
