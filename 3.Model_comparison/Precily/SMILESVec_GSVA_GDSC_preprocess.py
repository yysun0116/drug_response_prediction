import pandas as pd
import numpy as np
import pickle
import os

datadir = '/Users/yihyun/Code/'
# get list of molecules in GDSC dataset
GDSC = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/GDSC2_external_IC50.csv"), index_col=0)
with open(os.path.join(datadir, "drug_sensitivity_prediction/3.Model_comparison/Precily/GDSC_mol.txt"), "w") as nf:
    for mol in GDSC.index.to_list():
        nf.write(mol)
        nf.write("\n")
nf.close()

# get word embeddings of molecules in GDSC via SMILESVec model
## git clone https://github.com/hkmztrk/SMILESVecProteinRepresentation.git
## python3 getsmilesvec.py drug.l8.pubchem.canon.ws20.txt

# import processed word embeddings of molecules in GDSC dataset
with open(os.path.join(datadir, "drug_sensitivity_prediction/3.Model_comparison/Precily/GDSC_smiles.vec"), "rb") as f:
    GDSC_vec = pickle.loads(f.read())
GDSC_vec_df = pd.DataFrame(np.array(GDSC_vec), index = GDSC.index)

# apply GSVA on CCLE gene expression (but not work as in Precily paper)
# from GSVA import gsva, gmt_to_dataframe
# CCLE = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/raw/CCLE_expression.csv"), index_col = 0)
# CCLE.columns = [cl.split("(")[0].strip() for cl in CCLE.columns]

# genesets_df = gmt_to_dataframe(os.path.join(datadir, "drug_sensitivity_prediction/3.Model_comparison/Precily/MSigDB/c2.cp.v6.1.symbols.gmt'))
# genesets_df.head()
# pathways_df = gsva(np.log1p(CCLE.T),genesets_df, tempdir= os.path.join(datadir, "drug_sensitivity_prediction/3.Model_comparison/Precily/MSigDB/"))


# use cell lines in training and testing set of Precily(all testing cell lines are in training set of Precily) as testing cell lines to get their GSVA score vectors
## git clone https://github.com/SmritiChawla/Precily.git
precily_gsva = pd.read_csv(os.path.join(datadir, "repo/Precily/Data/Complete_Training_data_for_DNN.csv"))
precily_gsva_test = pd.read_csv(os.path.join(datadir, "repo/Precily/Fig1/Fig1d/Test_Set.csv"))
precily_gsva_clean = precily_gsva[["dt.merged$CELL_LINE_NAME"]+precily_gsva.columns[2:1331].to_list()].drop_duplicates().set_index("dt.merged$CELL_LINE_NAME")
precily_gsva_test_clean = precily_gsva_test[["0"]+precily_gsva_test.columns[2:1331].to_list()].drop_duplicates().set_index("0")
precily_gsva_test_clean.columns = precily_gsva_clean.columns.to_list()

## get cell line names from CCLE information data
CCLE_info = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/raw/CCLE_sample_info.csv"), index_col = 0)
cl_name = CCLE_info.loc[GDSC.columns]["stripped_cell_line_name"].values
## GSVA score vector: keep the intersection of training cell lines of Precily and GDSC
precily_gsva_clean_gdsc = precily_gsva_clean.loc[list(set(cl_name) & set(precily_gsva_clean.index))]
## GDSC dataset: keep the intersection of training cell lines of Precily and GDSC
GDSC_precily = GDSC[CCLE_info.loc[list(set(cl_name) & set(precily_gsva_clean.index))]["DepMap_ID"].values]
GDSC_precily =  GDSC_precily.reset_index()
GDSC_precily_long = pd.melt(GDSC_precily, id_vars = "index", value_vars = GDSC_precily.columns[1:]).dropna().reset_index()

# concat GSVA score vector and SMILESVec word embedding
path_vec_concat = np.concatenate([np.array(precily_gsva_clean.loc[GDSC_precily_long['cell_line']]), np.array(GDSC_vec_df.loc[GDSC_precily_long['index']])],1)
# rename column names as PATHWAY NAMEs and X1~X100
GDSC_X_df = pd.DataFrame(path_vec_concat, columns = precily_gsva_test_clean.columns.to_list() + ["X%d" %(num+1) for num in range(100)])
GDSC_X_df["cell_line"] = GDSC_precily_long['cell_line']
GDSC_X_df["drug_name"] = GDSC_precily_long['index']
GDSC_X_df["IC50"] = GDSC_precily_long['value']
GDSC_X_df.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/3.Model_comparison/Precily/Precily_GDSC_test.csv"), index = False)

# prediction
## run repo/Precily/Fig1/Fig1d/Fig1d.R