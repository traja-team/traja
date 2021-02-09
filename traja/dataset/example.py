import pandas as pd

default_cache_url = "dataset_cache"


def jaguar(cache_url=default_cache_url):
    # Sample data
    data_url = "https://raw.githubusercontent.com/traja-team/traja-research/dataset_und_notebooks/dataset_analysis/jaguar5.csv"
    df = pd.read_csv(data_url, error_bad_lines=False)
    return df
