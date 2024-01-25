# Drug Response Prediction using Gene Expression and Molecular Structure 
**This is the comprehensive source code for my Master's Thesis on Drug Response Prediction using Gene Expression and Molecular Structure**

Author:
- Sun Yih-Yun (孫懿筠)

## Outline
0. data  
    Including both the raw and preprocessed data along with the preprocessing process in this work
     - [PRISM](https://depmap.org/portal/download/all/)
     - [GDSC2](https://depmap.org/portal/download/all/)
1. Computational_method_comparison  
    Comparing the performance of different models for the task of drug response prediction (drug-blind testing: unknown compounds and known cell lines)
     - Matrix-Factorization model (MF_model)
     - Machine Learning model (ML_model)
     - NN model with [SMILESVec protein representation](https://github.com/hkmztrk/SMILESVecProteinRepresentation) (SMILESVec_model)
     - First-order Weisfeiler-Lehman GNN model (WL_GNN_model)
2. Combined_model_evaluation  
    Constructing a combined (2-step) model for the prediction of new drugs
     - CaDRReS_CLsim: Drug response prediction on known compounds and unknown cell lines based on the similarity of cell lines (prediction of cell-blind testing set)
     - CaDRReS_CLsim_SVM: Drug response prediction on unknown compounds and unknown cell lines based on the similarity of molecular structure (prediction of disjoint testing set)
3. Model_comparison  
    Comparing the performance of the combined model with [Precily](https://github.com/SmritiChawla/Precily/tree/main) on external testing data
4. User-friendly_interface  
    Developing a user-friendly drug response prediction tool using a Docker image
5. Discussion  
    - lineage_analysis
    - dose_range
    - parameter_tuning
  
## Environment
- System apps
  - Python 3.10.0+
- Python packages
  - requirements.txt
