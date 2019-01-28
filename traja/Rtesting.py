# import traja
# import pandas as pd
# try:
#     import rpy2
# except ImportError:
#     raise ImportError("""
#
#     Error: rpy2 not installed. Install it with
#
#     pip install rpy2
#
#     """)
# import rpy2.robjects.packages as rpackages
# import rpy2.robjects.numpy2ri
# import rpy2.robjects.pandas2ri as rpandas
#
# from rpy2.robjects.vectors import DataFrame
# from rpy2.robjects.packages import importr, data
#
# utils = rpackages.importr('utils')
# utils.install_packages('adehabitatLT')
# utils.chooseCRANmirror(ind=1) # select the first mirror in the list
# # base = importr('base')
# rpandas.activate()
# adehabitat = importr('adehabitatLT')
# df = pd.DataFrame({'x':[1,2,3,4,5],'y':[2,5,3,4,1]})
# rdf = rpandas.py2rpy(df)
# ltraj = adehabitat.as_ltraj(rdf, id=1, typeII=False)
# # df = df.dropna() # Adehabitat method requires no NANs
# # date_series = df.index.toseries()
# # date = base.as_POSIXct(date_series, format="%Y-%m-%d %H:%M:%OS")
#
# # xy = df[['x','y']]
#
# # Create ltraj object for adehabitat analysis and plotting.
# # ltraj = adehabitat.as_ltraj(xy, date=rdate, id='1')
#
