#!/usr/bin/env bash
script=$(dirname $0)/pascalvoc.py
python ${script} \
    --gtfolder /opt/aimledge/datasets/person-detection/annotations/\
    --detfolder /opt/aimledge/datasets/person-detection/voc-detections\
    --threshold 0.5\
    -gtformat xyrb\
    --gt_file_format xml\
    -detformat xyrb\
    --det_file_format txt\
    -gtcoords abs\
    -detcoords abs\
    --savepath ./det_eval_results\

