import pandas as pd
import numpy as np
import requests
import os
from clint.textui import progress

default_cache_url = 'dataset_cache'


def jaguar(cache_url=default_cache_url):
	# Sample data
	data_url = "https://raw.githubusercontent.com/traja-team/traja-research/dataset_und_notebooks/dataset_analysis/jaguar5.csv"
	df = pd.read_csv(data_url, error_bad_lines=False)
	return df


def Elk_in_southwestern_Alberta():

	"""
	URL:- https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study933711994

	Licence Terms:- https://creativecommons.org/publicdomain/zero/1.0/

	Citation:- Boyce MS and Ciuti S (2020) Data from: Human selection of elk behavioural traits in a landscape of fear. 
	Movebank Data Repository. <a href="https://www.doi.org/10.5441/001/1.j484vk24" target="_blank">https://www.doi.org/10.5441/001/1.j484vk24</a>
	Paton DG, Ciuti S, Quinn M, Boyce MS (2017) Hunting exacerbates the response to human disturbance in large herbivores while migrating through a road network. 
	Ecosphere 8(6): e01841. https://doi.org/10.1002/ecs2.1841
	Prokopenko CM, Boyce MS, Avgar T (2017) Characterizing wildlife behavioural responses to roads using integrated step selection analysis. Journal of Applied Ecology 54: 470-479. 
	https://doi.org/10.1111/1365-2664.12768 
	Prokopenko CM, Boyce MS, Avgar T (2017) Extent-dependent habitat selection in a migratory large herbivore: road avoidance across scales. Landscape Ecology 32: 313-325. https://doi.org/10.1007/s10980-016-0451-1 
	Roberts DR, Bahn V, Ciuti S, Boyce MS, Elith J, Guillera-Arroita G, Hauenstein S, Lahoz-Monfort JJ, Schöder B, Thuiller W, Warton DI, Wintle BA, Hartig F, Dormann CF (2017) Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. 
	Ecography 40: 913-929. https://doi.org/10.1111/ecog.02881 
	Thurfjell H, Ciuti S, Boyce MS (2017) Learning from the mistakes of others: How female elk (Cervus elaphus) adjust behaviour with age to avoid hunters. PLoS ONE 12(6): e0178082. 
	https://doi.org/10.1371/journal.pone.0178082 
	Benz RA, Boyce MS, Thurfjell H, Paton DG, Musiani M, Dormann CF, et al. (2016) Dispersal ecology informs design of large-scale wildlife corridors. PLoS ONE 11(9): e0162989. https://doi.org/10.1371/journal.pone.0162989 
	Ensing EP, Ciuti S, de Wijs FALM, Lentferink DH, ten Hoedt A, Boyce MS, Hut RA (2014) GPS based daily activity patterns in European red deer and North American elk (Cervus elaphus): indication for a weak circadian clock in ungulates. 
	PLoS ONE 9(9): e106997. https://doi.org/10.1371/journal.pone.0106997 
	Killeen J, Thurfjell H, Ciuti S, Paton D, Musiani M, Boyce MS (2014) Habitat selection during ungulate dispersal and exploratory movement at broad and fine scale with implications for conservation management. 
	Movement Ecology 2:15. https://doi.org/10.1186/s40462-014-0015-4
	Thurfjell H, Ciuti S, Boyce MS (2014) Applications of step-selection functions in ecology and conservation. 
	Movement Ecology 2:4. https://doi.org/10.1186/2051-3933-2-4 
	Ciuti S, Muhly TB, Paton DG, McDevitt AD, Musiani M, Boyce MS (2012) Human selection of elk behavioural traits in a landscape of fear. 
	Proceedings of the Royal Society B 279(1746): 4407-4416. https://doi.org/10.1098/rspb.2012.1483 
	Ciuti S, Northrup JM, Muhly TB, Simi S, Musiani M, Pitt JA, Boyce MS (2012) Effects of humans on behaviour of wildlife exceed those of natural predators in a landscape of fear. 
	PLoS ONE 7(11): e50611. https://doi.org/10.1371/journal.pone.0050611

	Principal Investigator Name: Mark S. Boyce
	"""

	csv_folder = os.path.join("traja", "dataset", "CSVs")
	store_csv_here = os.path.join(csv_folder, "Elk_in_southwestern_Alberta.csv")
	
	if not os.path.exists(csv_folder):
		os.makedirs(csv_folder)

	if not os.path.exists(store_csv_here):
		url = "https://traja-datasets.s3.eu-central-1.amazonaws.com/movebank/Elk-in-southwestern-Alberta/Elk_in_southwestern_Alberta.csv"
		response = requests.get(url, stream=True)
		with open(store_csv_here, 'wb') as f:
			total_length = int(response.headers.get('content-length'))
			for chunk in progress.bar(response.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
				if chunk:
					f.write(chunk)
					f.flush()
	
	df = pd.read_csv(store_csv_here,low_memory=False)

	unique_names = df["individual-local-identifier"].unique()
	df = df[['location-long', 'location-lat', 'individual-local-identifier']]
	df.rename(columns={'location-long': 'x', 'location-lat': 'y', 'individual-local-identifier': 'ID'}, inplace=True)

	old_dict = dict(enumerate(unique_names.flatten(), 1)) 
	new_dict = dict([(value, key) for key, value in old_dict.items()]) 
	id_series = [new_dict[x] for x in df['ID']]

	df['ID'] = pd.Series(id_series)
	return df