import pandas as pd
import numpy as np
import pubchempy as pcp
import pickle
import rdkit
from rdkit import Chem
import os

datadir = '/Users/yihyun/Code/'
# import data
## drug response data
GDSC = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/raw/sanger-dose-response.csv"))
## drug information data
GDSC_compound =  pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/raw/screened_compounds_rel_8.4.csv"), index_col="DRUG_ID")

# filter out the data from GDSC2
GDSC2 = GDSC[GDSC['DATASET'] == "GDSC2"]

# drug ID
drug_ids = np.unique(GDSC2['DRUG_ID'].tolist())
# drug name
identifiers = np.unique(GDSC2['DRUG_NAME'].tolist())
# get all drug names from drug information data
GDSC_all_compounds = list(GDSC_compound.loc[set(drug_ids) - set([1807, 1818]), 'DRUG_NAME']) + ['VINCRISTINE', 'CARMUSTINE']

# get SMILES string
smiles_dict = {}
not_found = []
for ids in drug_ids:
    if ids in GDSC_compound.index.tolist():
        drug_name = GDSC_compound.loc[ids, 'DRUG_NAME']
    else:
        drug_name = GDSC2.loc[GDSC2['DRUG_ID'] == ids, 'DRUG_NAME'].tolist()[0]
        drug_name = drug_name.split(",")[0]

    try:
        results = pcp.get_compounds(drug_name, 'name')
        smiles_dict[ids] = results[0].canonical_smiles
        print(drug_name, smiles_dict[ids])
    #except IndexError:
    #    try:
    #        results = pcp.Compound.from_cid(int(ids))
    #        smiles_dict[ids] =  results.canonical_smiles
    #        print(ids, smiles_dict[ids])
        #except ValueError:
            #print(ids, 'is not found in PubChem')
    except IndexError:
        not_found.append(ids)
        print(ids, 'is not found in PubChem')

## check molecules whose SMILES string was not found by pcp
not_found # [1047, 1635, 1855, 1866, 1998]
## manual search online (molecule name)
### 1047
results = pcp.get_compounds("Nutlin-3a", 'name')
smiles_dict[1047] = results[0].canonical_smiles
### 1635
results = pcp.get_compounds("Picolinate", 'name')
smiles_dict[1635] = results[0].canonical_smiles
### 1855
results = pcp.get_compounds("GTPL8020", 'name')
smiles_dict[1855] = results[0].canonical_smiles

with open("/volume/yihyun/drug/GDSC/GDSC_smiles.pickle", 'wb') as nf:
    pickle.dump(smiles_dict, nf)

# data cleaning
## keep data with SMILES string found in previos step
GDSC2['smiles'] = [smiles_dict[id] if id in smiles_dict.keys() else np.nan for id in GDSC2['DRUG_ID']]
GDSC2_cleaned = GDSC2[["DRUG_ID", "smiles", "ARXSPAN_ID", "IC50_PUBLISHED", "AUC_PUBLISHED"]].dropna()
## check if there are duplicates in dataset
np.unique(GDSC2_cleaned.loc[GDSC2_cleaned.duplicated(["smiles", "ARXSPAN_ID"], keep = 'last'), "DRUG_ID"])
### [1007, 1089, 1200, 1553, 1811, 1908]
## take the average of the duplicates
GDSC2_cleaned.groupby(['smiles','ARXSPAN_ID'], as_index=False).mean() # long data

## wide data
### IC50 as measurement
GDSC2_cleaned_wide = pd.pivot(GDSC2_cleaned.groupby(['smiles','ARXSPAN_ID'], as_index=False).mean(),
                              index = "smiles", columns = "ARXSPAN_ID", values="IC50_PUBLISHED")
GDSC2_cleaned_wide.to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/GDSC_cleaned_IC50.csv"), index=True)
### AUC as measurement
GDSC2_cleaned_wide = pd.pivot(GDSC2_cleaned.groupby(['smiles','ARXSPAN_ID'], as_index=False).mean(),
                              index = "smiles", columns = "ARXSPAN_ID", values="AUC_PUBLISHED")
GDSC2_cleaned_wide.to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/GDSC_cleaned_auc.csv"), index=True)

## get canonical SMILES
GDSC_cs = []
for smiles in GDSC2_cleaned_wide.index.tolist():
    molecule = Chem.MolFromSmiles(smiles)
    #molecule = Chem.AddHs(molecule)
    canonical_smiles = Chem.MolToSmiles(molecule)
    GDSC_cs.append(canonical_smiles)
## set canonical SMILES of the molecules as index of GDSC2 
GDSC2_cleaned_wide.index = GDSC_cs

# import PRISM dataset
prism_train = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_train.csv"), index_col = 'smiles')
prism_test = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_test.csv"), index_col = 'smiles')
prism_test_cl = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_test_scdrug.csv"), index_col = 'smiles')
prism_train_smiles = prism_train.index.tolist()
prism_test_smiles = prism_test.index.tolist()
## get molecules in PRISM
prism_smiles = prism_train.index.tolist() + prism_test.index.tolist()
## get canonical SMILES of molecules in PRISM (ALL)
PRISM_cs = []
for smiles in prism_smiles:
    molecule = Chem.MolFromSmiles(smiles)
    #molecule = Chem.AddHs(molecule)
    canonical_smiles = Chem.MolToSmiles(molecule)
    PRISM_cs.append(canonical_smiles)

# keep the moleules and cell lines not in PRISM as external testing set
GDSC2_external = GDSC2_cleaned_wide.loc[list(set(GDSC_cs).difference(set(PRISM_cs))), 
                       list(set(GDSC2_cleaned_wide.columns).difference(set(prism_train.columns.tolist() + prism_test_cl.columns.tolist())))]
# save data
GDSC2_external.to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/GDSC2_external_auc.csv"), index = True)



