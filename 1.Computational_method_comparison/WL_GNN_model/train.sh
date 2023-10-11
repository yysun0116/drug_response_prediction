#!/bin/bash

file_dir=/Users/yihyun/Code/Drug_sensitivity_prediction/0.data/raw
task=regression  # target is a real value (e.g., energy eV).
dataset=PRISM

radius=1
dim=50
layer_hidden=6
layer_output=6

batch_train=32
batch_test=32
lr=1e-4
lr_decay=0.99
decay_interval=10
iteration=1000


setting=$dataset--radius$radius--dim$dim--layer_hidden$layer_hidden--layer_output$layer_output--batch_train$batch_train--batch_test$batch_test--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--iteration$iteration
python3.8 -u /Users/yihyun/Code/Drug_sensitivity_prediction/1.Computational_method_comparison/WL_GNN_model/MolecularGNN_smiles_train.py $file_dir $task $dataset $radius $dim $layer_hidden $layer_output $batch_train $batch_test $lr $lr_decay $decay_interval $weight_decay $iteration $setting

