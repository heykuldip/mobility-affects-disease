"""Fig. 2: how Advan-based destination weights are distributed across places.

    python analysis/destination_weights.py <advan_pois.csv> <buildings.csv> [--out visitors_distribution.pdf]

<advan_pois.csv>: Advan places matched to buildings, with bldg_id, bldg_type and home_cbgs (visitor counts by home
census block group). <buildings.csv>: the simulation's buildings, with GEOID10 (census tract). Advan data are
licensed, so neither file is included in this repository.

For each place type, a place's weight for a home tract is its share of the tract's Advan visits to places of that
type, as in the Advan-based models. Each place's weights are averaged over all home tracts (a tract with no visits
counts as zero), and the five place types are weighted equally.
"""
import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLACE_TYPES = ['errands', 'restaurants', 'hospitals', 'recreation', 'schools']

p = argparse.ArgumentParser()
p.add_argument('advan_pois')
p.add_argument('buildings')
p.add_argument('--out', default='visitors_distribution.pdf')
a = p.parse_args()

pois = pd.read_csv(a.advan_pois, usecols=['bldg_id', 'bldg_type', 'home_cbgs'])
tracts = set(pd.read_csv(a.buildings, usecols=['GEOID10'], dtype={'GEOID10': str})['GEOID10'].dropna())

weights = []
for place_type in PLACE_TYPES:
    sel = pois[pois['bldg_type'] == place_type]
    visits = pd.DataFrame([(cbg[:11], b, n) for b, cbgs in zip(sel['bldg_id'], sel['home_cbgs'])
                           for cbg, n in ast.literal_eval(cbgs).items()],
                          columns=['census_tract', 'bldg_id', 'n_visits'])
    visits = visits[visits['census_tract'].isin(tracts)]
    visits = visits.groupby(['census_tract', 'bldg_id'])['n_visits'].sum().reset_index()
    visits['weight'] = visits['n_visits'] / visits.groupby('census_tract')['n_visits'].transform('sum')
    weights.append(visits)

mean_weight = pd.concat(weights).groupby('bldg_id')['weight'].sum() / len(tracts)
prob = np.sort((mean_weight / mean_weight.sum()).to_numpy())[::-1]

plt.loglog(np.arange(1, len(prob) + 1), prob, marker='.')
plt.xlabel('Places ranked by Advan-based weights')
plt.ylabel('Selection Probability')
plt.savefig(a.out, dpi=600, format='pdf', bbox_inches='tight')

k1, k10 = max(1, len(prob) // 100), max(1, len(prob) // 10)
print(f'{len(prob)} places | top 1%: {prob[:k1].sum():.2f} | top 10%: {prob[:k10].sum():.2f}')
