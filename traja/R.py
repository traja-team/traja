import traja
import pandas as pd
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri as rpandas
rpandas.activate()
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data

utils = rpackages.importr('utils')
utils.install_packages('adehabitatLT')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
base = importr('base')
adehabitat = importr('adehabitatLT')

# df = df.dropna() # Adehabitat method requires no NANs
# date_series = df.index.toseries()
# date = base.as_POSIXct(date_series, format="%Y-%m-%d %H:%M:%OS")

# !?Fractional seconds ignored for some reason (see https://bitbucket.org/rpy2/rpy2/issues/508/milliseconds-lost-converting-from-pandas)
# Initialize dataframe with millisecond precision
data = pd.DataFrame({
    'Timestamp': pd.date_range('2017-01-01 00:00:00.234', periods=20, freq='ms', tz='UTC')
    })

rdata = rpandas.py2ri(data.Timestamp)

## Example
# input:
# rdata[0]
# >> 1483228800.0
# input:
# rdata[1]
# >> 1483228800.0 # Should be different than previous timestep but loses milliseconds

# xy = df[['x','y']]

# Create ltraj object for adehabitat analysis and plotting.
# ltraj = adehabitat.as_ltraj(xy, date=rdate, id='1')
