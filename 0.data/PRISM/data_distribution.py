# import modules
import pandas as pd
import os

datadir = '/Users/yihyun/Code/'

train_set = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_train.csv"), index_col="smiles")
test_set = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_test.csv"), index_col="smiles")
# label the data by the division 
train_set["set"] = "train"
test_set["set"] = "test"

# concatenate the data
frames = [train_set, test_set]
result = pd.concat(frames)

# PCA
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

np.random.seed(1234)
pca = PCA(n_components=2)
X = result.fillna(0)[result.columns[:-1]]
newX = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
### pca plot
px.scatter(newX, x=0, y=1, color=result['set'])

# TSNE
from sklearn.manifold import TSNE
import seaborn as sns

np.random.seed(1234)
rndperm = np.random.permutation(test_set.shape[0])
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

### TSNE plot
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue="set",
    palette=sns.color_palette("hls", 10),
    data=result,
    legend="full",
    alpha=0.3
)
# outlier: Clc1ccc(CSC(Cn2ccnc2)c2ccc(Cl)cc2Cl)cc1

# exclude the outlier from the data
train_df_exclude = train_set[train_set["smiles"] != "Clc1ccc(CSC(Cn2ccnc2)c2ccc(Cl)cc2Cl)cc1"]
train_df_exclude["set"] = "train"
train_df_exclude = train_df_exclude.set_index("smiles")
# delete the label column
train_df_exclude[train_df_exclude.columns[:-1]]
# save the data excluded the outlier
train_df_exclude[train_df_exclude.columns[:-1]].to_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_train_exclude.csv"), index = True)


