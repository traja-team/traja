""" Yanked from https://heremaps.github.io/pptk/tutorials/viewer/geolife.html"""

import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os

import traja

__all__ = ["read_plt", "read_labels", "read_all_users", "apply_labels", "load_geolife"]

def read_plt(plt_file):
    points = pd.read_csv(plt_file, skiprows=6, header=None,
                         parse_dates=[[5, 6]], infer_datetime_format=True)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])

    return points

mode_names = ['walk', 'bike', 'bus', 'car', 'subway','train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s : i + 1 for i, s in enumerate(mode_names)}

def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

    # for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

    # replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels

def apply_labels(points, labels):
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points['label'][no_label] = 0

def read_user(user_folder):
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    df = pd.concat([read_plt(f) for f in plt_files])

    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['label'] = 0

    return df

def read_all_users(folder):
    subfolders = os.listdir(folder)
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder,sf))
        df['user'] = int(sf)
        dfs.append(df)
    return pd.concat(dfs)

def load_geolife(folder: str, as_traja=True, lat=(32, 48.0), lon=(114, 120)):
    """Read Microsoft Geolife data from folder. Default mask in UTM Zone 50 (Beijing)
    
    Args:
        folder (str): path to Geolife dataset (https://www.microsoft.com/en-us/download/details.aspx?id=52367)
        as_traja (bool): create x and y columns from latitude/longitude
        lat (tuple): latitude mask
        lon (tuple): longitude mask
    
    Returns:
        pandas or traja DataFrame
        
    """
    try:
        import pyproj
    except ImportError:
        raise ImportError(
            """Mising pyproj
            Please download it with pip install pyproj
    """
        )
    
    df = traja.dataset.geolife.read_all_users(folder)
    if as_traja:
        # Convert lat/long to utm coordinates
        if lat and lon:
            geomask = (
                (df["lon"] > lon[0])
                & (df["lon"] < lon[1])
                & (df["lat"] > lat[0])
                & (df["lat"] < lat[1])
            )
            df = df[geomask]
        proj = pyproj.Proj(proj='utm', zone=50, ellps='WGS84')
        x, y = proj(df["lon"].tolist(), df["lat"].tolist())
        df["x"] = x
        df["y"] = y

    return df