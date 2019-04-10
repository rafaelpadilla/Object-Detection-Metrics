#!/usr/bin/env bash
script=$(dirname $0)/eval.py
python ${script} \
    --gt_dir ${1}\
    --det_dir ${2}\
    --threshold 0.5\
    -gt_format xyrb\
    --gt_file_format xml\
    -det_format xyrb\
    --det_file_format txt\
    -gt_coords abs\
    -det_coords abs\

