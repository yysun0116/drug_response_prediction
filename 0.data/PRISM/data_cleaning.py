import numpy as np
import pandas as pd
import timeit
import os

datadir = "/Users/yihyun/Code/"

# import PRISM data
print("import PRISM data...")
prism = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/raw/secondary-screen-dose-response-curve-parameters.csv"))

### 將AUC大於1者限制於1
print("bound the auc value to 1")
for i in range(len(prism["auc"])):
    if prism["auc"][i] > 1:
        prism["auc"][i] = float(1)

### 將資料根據screen_id由大到小排序(後續去除duplicate時才會留下screen_id較大者)
prism_sorted = prism.sort_values(by=["screen_id"], ascending=False)
prism_sorted = prism_sorted.drop_duplicates(subset=["depmap_id","name"])
print("PRISM data size after dropping out the drug with more than 1 screening:",len(prism_sorted))

prism_small = prism_sorted[["name", "smiles", "auc", "depmap_id"]]

prism_small= prism_small.sort_values(by = "depmap_id", ascending=False)
### 去除含有NA值的資料 (STR profiling test failed)
prism_clean = prism_small.dropna()
print("PRISM data size after dropping out the data failed in STR profiling test:",len(prism_clean))

drug_name = np.unique(prism_clean["name"])
print("number of molecules in PRISM:",len(drug_name))
cell_line = np.unique(prism_clean["depmap_id"])
print("number of cell lines in PRISM:",len(cell_line))


# import CCLE data
print("import CCLE data...")
ccle = pd.read_csv(os.path.join(datadir, "raw", "CCLE_expression.csv"))
# rename column 0 with "cell_lines"
colnames = ccle.columns
colnames = np.array(colnames)
colnames[0] = "cell_lines"
ccle.columns = colnames

# 找出prism 與 ccle交集的cell line
PRISM_intersect_ccle = set(ccle["cell_lines"]) & set(cell_line)
PRISM_intersect_ccle = sorted(PRISM_intersect_ccle)
len(PRISM_intersect_ccle)

### 去除PRISM中不在CCLE data中的cell lines
prism_inter_ccle = prism_clean[(prism_clean["depmap_id"]).isin(PRISM_intersect_ccle)]
len(prism_inter_ccle)
# save prism data as csv
prism_inter_ccle = prism_inter_ccle.reset_index()
# 將多smiles欄位有多於一個者指定為最後一個
for idx in range(len(prism_inter_ccle["smiles"])):
    prism_inter_ccle["smiles"][idx] = prism_inter_ccle.loc[idx]["smiles"].split(",")[-1].strip()
prism_inter_ccle.to_csv(os.path.join(datadir, "processed","prism_inter_ccle.csv", index=False))

### CCLE僅保留PRISM和CCLE共有的cell lines
ccle_inter_mat = ccle[ccle["cell_lines"].isin(PRISM_intersect_ccle)]
ccle_inter_mat_sorted = ccle_inter_mat.sort_values(by="cell_lines", ascending=True)
ccle_inter_mat_sorted.set_index("cell_lines" , inplace=True)
# 去除在ccle data columns (gene names)名稱後面多餘的括號
gene_ls = []
for gene in ccle_inter_mat_sorted.columns.tolist():
    gene_ls.append(gene.split(" (")[0])
ccle_inter_mat_sorted.columns = gene_ls
ccle_inter_mat_sorted.to_csv(os.path.join(datadir, "processed","ccle_inter_mat_sorted.csv", index=True))

