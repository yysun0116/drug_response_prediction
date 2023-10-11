#!/bin/bash
set -e

# Required

if [ -z $IN_FILE ]; then
    echo "Err: missing input parameters"
    exit 1
fi

if [ -z $OUT_DIR ] ; then
    echo "Err: missing output parameters"
    exit 2
fi

# Resource
mkdir -p /opt/CaDRReS-Sc/data/CCLE
mkdir -p /opt/CaDRReS-Sc/preprocessed_data/PRISM

if [ -z $REF_CCLE ]; then
    echo "Err: missing CCLE expression file"
    exit 4
else
    ln -s $REF_CCLE /opt/CaDRReS-Sc/data/CCLE/CCLE_expression.csv
fi

if [ -z $REF_GDSC ] ; then
    echo "Err: missing GDSC expression file"
    exit 4
else
    ln -s $REF_GDSC /opt/CaDRReS-Sc/data/GDSC/GDSC_exp.tsv
fi

if [ -z $REF_FEATURE ] ; then
    echo "Err: missing feature information for PRISM"
    exit 4
else
    ln -s $REF_FEATURE /opt/CaDRReS-Sc/preprocessed_data/PRISM/feature_genes.txt
fi

if [ -z $REF_DRUG_INFO ] ; then
    echo "Err: missing drug information for PRISM"
    exit 4
else
    ln -s $REF_DRUG_INFO /opt/CaDRReS-Sc/preprocessed_data/PRISM/PRISM_drug_info.csv
fi

# Make script

mkdir -p $OUT_DIR
echo -n "python3 /scDrug/script/drug_response_prediction.py -o ${OUT_DIR} -i ${IN_FILE} --n_drugs ${N_DRUGS}" >> tmp.sh

# Optional

if [ ! ${CLUSTERS} = "None" ]; then
    echo -n " -c ${CLUSTERS} " >> tmp.sh
fi

if [ ${MODEL} = "PRISM" ] || [ ${MODEL} = "GDSC" ]; then
    echo -n " -m ${MODEL} " >> tmp.sh
else
    echo "Err: wrong model name"
    exit 3
fi


## drug discovery option
if [ ${DRUG_DISCOVERY} = "FALSE" ]; then
   echo "not include drug discovery process"
else
   if [ -z $NEW_DRUG ]; then
      echo "Err: missing new drug smiles file"
      exit 4
   else
      echo -n " && python3 /scDrug/script/new_drug_prediction.py -i ${OUT_DIR} -smiles ${NEW_DRUG} -o ${OUT_DIR} -m ${MODEL}" >> tmp.sh
   fi
fi



# Execute
/usr/bin/time -f "scDrug sensitivity prediction mem=%K RSS=%M elapsed=%E cpu.sys=%S .user=%U" \
    bash tmp.sh
rm tmp.sh
