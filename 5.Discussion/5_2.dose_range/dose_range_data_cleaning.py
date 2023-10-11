import numpy as np
import pandas as pd
import os

datadir = '/Users/yihyun/Code'
# cell viability data
prism_CV = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/raw/secondary-screen-replicate-collapsed-treatment-cell-viability-merge-dose.csv"), index_col=0)
prism_CV_info = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/raw/secondary-screen-replicate-collapsed-treatment-info.csv"), index_col=0)

# check different batches in PRISM
np.unique(prism_CV_info['screen_id'])

prism_cv_collapsed_info_clean = prism_CV_info.dropna(subset = ["smiles"])
## some molecules with more than one smiles encoding -> leave the last one
for i in range(len(prism_cv_collapsed_info_clean)):
    prism_cv_collapsed_info_clean['smiles'][i] = prism_cv_collapsed_info_clean['smiles'][i].split(", ")[-1]


# calculate dose range
dose_range_allBatch = []
for batch in np.unique(prism_CV_info['screen_id']):
    batch_data = prism_cv_collapsed_info_clean[["smiles", "dose"]][prism_cv_collapsed_info_clean['screen_id'] == batch]
    batch_dose_range = np.empty((len(np.unique(batch_data['smiles'].values)), 3))
    batch_dose_range[:] = np.nan
    batch_dose_range_df = pd.DataFrame(batch_dose_range, columns=["batch","smiles", "range"])
    batch_dose_range_df['batch'] = batch
    print(batch)
    for idx in range(len(np.unique(batch_data['smiles']))):
        mol = np.unique(batch_data['smiles'])[idx]
        tmp_data = batch_data["dose"][batch_data['smiles']== mol]
        if len(tmp_data) == 8:
            batch_dose_range_df.loc[idx, 'smiles'] = mol
            batch_dose_range_df.loc[idx, 'range'] = max(tmp_data) - min(tmp_data)
        elif len(tmp_data) > 16:
            print(mol)
            batch_dose_range_df.loc[idx, 'smiles'] = mol
            batch_dose_range_df.loc[idx, 'range'] = max(tmp_data) - min(tmp_data)
        else:
            print(mol)
            batch_dose_range_df.loc[idx, 'smiles'] = mol
            batch_dose_range_df.loc[idx, 'range'] = max(tmp_data[-8:]) - min(tmp_data[-8:])
    dose_range_allBatch.append(batch_dose_range_df)

allbatch_dose_range_merge = pd.merge(dose_range_allBatch[3], dose_range_allBatch[2], left_index=False, right_index=False, how='outer')
allbatch_dose_range_merge = pd.merge(allbatch_dose_range_merge, dose_range_allBatch[1], left_index=False, right_index=False, how='outer')
allbatch_dose_range_merge = pd.merge(allbatch_dose_range_merge, dose_range_allBatch[0], left_index=False, right_index=False, how='outer')

allbatch_dose_range_noduplicate = allbatch_dose_range_merge.drop_duplicates(subset = 'smiles')
range_mol = allbatch_dose_range_noduplicate["smiles"][allbatch_dose_range_noduplicate['range'] > 9.999].values


with open(os.path.join(datadir, "drug_response_prediction/5.Discussion/5_2.dose_range/dose_range_mol.txt"), "w") as nf:
    for smiles in range_mol:
        nf.write(smiles)
        nf.write("\n")
nf.close()

# visualization
import altair as alt

#tmp_data = HTS002_data_clean
#tmp_data = MTS005_data_clean
#tmp_data = MTS006_data_clean
#tmp_data = MTS010_data_clean

tmp_data = pd.melt(allbatch_dose_range_noduplicate, value_vars=allbatch_dose_range_noduplicate.columns.to_list()[1:], id_vars='smiles')
tmp_data.columns = ['smiles', 'label', 'dose']

alt.data_transformers.enable('default', max_rows=None)
chart = alt.Chart(
    data=tmp_data
).transform_filter(
    filter={"field": 'smiles',
            "oneOf": list(np.unique(tmp_data['smiles']))}
).transform_filter(
    filter={'field': 'label',
            "oneOf": np.unique(tmp_data['label'])}
)
line = chart.mark_line().encode(
    x='dose',
    y='smiles',
    detail='smiles:N'
)
points = chart.mark_point(
    size=15,
    opacity=1,
    filled=True
).encode(
    x='dose:Q',
    y='smiles:N'
)
(line + points)


