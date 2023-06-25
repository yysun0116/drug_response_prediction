import pandas as pd
import numpy as np
import os
import random

datadir = "/Users/yihyun/Code/"

# import clean prism data
prism_inter_ccle = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/raw/prism_inter_ccle.csv"))
# long data to wide data
prism_inter_ccle_wide = pd.pivot(prism_inter_ccle, index="smiles", columns="depmap_id", values="auc")

# convert auc to 1-auc
df = 1-prism_inter_ccle_wide
mask = (df.isna() == False).astype(int)
#df = df.fillna(value = -9)


# import testing drug list (10%, n = 144)
#with open("/volume/yihyun/drug/MolecularGNN_smiles/main/drug_list.txt", "r") as f:
#    drug_all_ls = f.read().split("\n")
drug_all_ls = list(prism_inter_ccle_wide.index)
np.random.seed(1234)
test_mol = random.sample(drug_all_ls, round(len(drug_all_ls) * 0.1))
train_mol = np.array(set(drug_all_ls) - set(test_mol))

# import testing cell line (for scDrug, n = 24)
testing_cl = np.array(["ACH-000123", "ACH-000189", "ACH-000209", "ACH-000228", "ACH-000244", "ACH-000252", "ACH-000288", "ACH-000329",
                   "ACH-000367", "ACH-000397", "ACH-000415", "ACH-000423", "ACH-000553", "ACH-000565", "ACH-000672", "ACH-000713", "ACH-000717",
                   "ACH-000764",  "ACH-000791", "ACH-000834", "ACH-000875", "ACH-000927", "ACH-000977", "ACH-001190"])
training_cl = np.array(set(df.columns) - set(testing_cl))

train_df = df.loc[train_mol][training_cl]
train_mask = mask.loc[train_mol][training_cl]
test_df = df.loc[test_mol][training_cl]
test_mask = mask.loc[test_mol][training_cl]
test_scDrug = df.loc[train_mol][testing_cl]
test_molGNN_scDrug = df.loc[test_mol][testing_cl]

train_df.to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_train.csv"), index = True)
test_scDrug.to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_test_scdrug.csv"), index = True)
test_df.to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_test.csv"), index = True)
test_molGNN_scDrug.to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_test_molGNN_scdrug.csv"), index = True)


# long data
## train
cl_list = list(train_df.columns)
train_df_long = pd.melt(train_df, id_vars="smiles", value_vars=cl_list).dropna()
with open(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_train_long.txt"), "w") as nf:
    for i in range(len(train_df_long)):
        tmp = list(train_df_long.loc[i])
        _, smiles, cl, auc = map(str, tmp)
        nf.write(" ".join([smiles, auc, cl]))
        nf.write("\n")
nf.close()
## test
cl_list = list(test_df.columns)
test_df_long = pd.melt(test_df, id_vars="smiles", value_vars=cl_list).dropna()
test_df_long = test_df_long.reset_index()
with open(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_test_long.txt"), "w") as nf:
    for i in range(len(test_df_long)):
        tmp = list(test_df_long.loc[i])
        _, smiles, cl, auc = map(str, tmp)
        nf.write(" ".join([smiles, auc, cl]))
        nf.write("\n")
nf.close()


