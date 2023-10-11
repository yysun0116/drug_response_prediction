#testing set kernel
import rdkit
from rdkit import DataStructs, Chem
import time
import numpy as np
from scipy import stats
import pandas as pd
from CaDRReS_DrugSim import load_model, predict_from_model

prism_test = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_drugBlind_test.csv", index_col="smiles")
prism_train = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_train.csv", index_col="smiles")

test_mol = prism_train.index.tolist() + prism_test.index.tolist()
test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol))
fps = [Chem.RDKFingerprint(x) for x in test_mol_rdkit]

start = time.time()
mol_sim_mat = np.zeros((len(test_mol), len(test_mol)))
for i in range(len(test_mol)):
    if (i+1)%100 == 0:
        print ("{} of {} ({:.2f})s".format(i+1, len(test_mol), time.time()-start))
        start = time.time()
    for j in range(len(test_mol)):
        sim_score = DataStructs.FingerprintSimilarity(fps[i], fps[j])
        mol_sim_mat[i, j] = sim_score

mol_sim_df = pd.DataFrame(mol_sim_mat, columns=test_mol, index=test_mol)
model_spec_name = "cadrres-wo-sample-bias_drugSim"
model_dir = "/volume/yihyun/drug/MF_model/result/"
model_file = model_dir + '{}_param_dict.pickle'.format(model_spec_name)
cadrres_model = load_model(model_file)

print('Predicting drug response using CaDRReS: {}'.format(model_spec_name))
pred_df, P_test_df= predict_from_model(cadrres_model, mol_sim_df, "cadrres-wo-sample-bias")
print('done!')

pred_df.loc[prism_test.index.tolist()].to_csv("/volume/yihyun/drug/MF_model/CaDRReS_DrugSim_pred.csv",index=True)