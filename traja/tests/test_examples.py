import os
import requests
import pandas as pd
import io
import hashlib

def test_Elk_in_southwestern_Alberta():
    url = "https://traja-datasets.s3.eu-central-1.amazonaws.com/movebank/Elk-in-southwestern-Alberta/Elk_in_southwestern_Alberta.csv"
    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("OOps: Something Else", err)

    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    hash_original = "1791c3919d6dabf593c3b328cd2ebd9d7a96f1cc2321b299f0c68c176725626b"
    sha = hashlib.sha256()
    sha.update(r.content)
    hash_downloaded = sha.hexdigest()

    assert len(df) == 876925
    assert hash_downloaded == hash_original

