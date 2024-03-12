#!/bin/bash
# Script: download_and_unzip.sh
# Description: Download and unzip KITTI dataset calibration and label files
# Author: [Your Name]
# Date: [Date]

CUR_DIR="$PWD"
cd data/KITTI

# Create directory if it doesn't exist
mkdir -p temp

# # Download calibration zip file
# curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip -o temp/data_object_calib.zip

# # Download label zip file
# curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip -o temp/data_object_label_2.zip

# Unzip calibration files
unzip -q temp/data_object_calib.zip -d temp/data_object_calib

# Unzip label files
unzip -q temp/data_object_label_2.zip -d temp/data_object_label_2

echo "Download and unzip complete."

cd $CUR_DIR
