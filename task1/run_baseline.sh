#!/bin/bash

python3 baseline/baseline.py -trp data/$1/train_$2.tsv -ttp data/$1/dev_test_$2.tsv
python3 scorer/evaluate.py -g data/$1/dev_test_$2.tsv -p data/$1/baseline_pred.tsv