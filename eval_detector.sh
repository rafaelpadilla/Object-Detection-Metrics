#!/usr/bin/env bash
script=$(dirname $0)/pascalvoc.py
python ${script} \
    --gtfolder ${1}\
    --detfolder ${2}\
    --threshold 0.5\
    -gtformat xyrb\
    --gt_file_format xml\
    -detformat xyrb\
    --det_file_format txt\
    -gtcoords abs\
    -detcoords abs\
    --savepath ./results\

