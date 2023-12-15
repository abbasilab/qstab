#!/bin/bash

## Tuning the distance sampling hyperparameter
PERTURB=True
PERTURB_SAMP="distance"
GROUP="drugs"
NDISTS="-50 -40 -30 -20 -10 -5 5 10 20 30 40 50 60"
NUMQ=6000
PYTHONPATH="..."
CODEPATH="./scripts/run_medalpaca_entity.py"
FPATH="./scripts/summary/medalpaca_usmle_drugs.txt"

echo "Running QA experiment on USMLE dataset with ..."
for NDIST in $NDISTS
do
    echo "perturb = $PERTURB, sampling = $PERTURB_SAMP, semgroup = $GROUP, numdistance = $NDIST, numquestion = $NUMQ" >> $FPATH
    $PYTHONPATH $CODEPATH -ptb=$PERTURB -ptb_samp=$PERTURB_SAMP -grp=$GROUP -ndist=$NDIST -nq=$NUMQ >> $FPATH
    echo "" >> $FPATH
done