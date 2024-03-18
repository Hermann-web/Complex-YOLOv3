#!/bin/bash
# Script: kitti_data_prep.sh
# Description: Prepare KITTI dataset for object detection
# Author: Hermann Agossou
# Date: [Date]

CUR_DIR="$PWD"
cd data/KITTI

# Set directories from disk
KITTI_OBJ_AND_IMG_DIR="/media/ubuntu/HBCD_PE_X64/data/Kitti/2011_09_26_drive_0106_sync"
KITTI_CALIB_DIR="/home/ubuntu/Documents/data/kitti/data_object_calib/training/calib"
KITTI_LABEL_DIR="/home/ubuntu/Documents/data/kitti/data_object_label_2/training/label_2"
KITTI_OBJ_DIR="$KITTI_OBJ_AND_IMG_DIR/velodyne_points/data"
KITTI_IMG_DIR="$KITTI_OBJ_AND_IMG_DIR/image_02/data"

# Set directories from temp
KITTI_OBJ_AND_IMG_DIR="/media/ubuntu/HBCD_PE_X64/data/Kitti/2011_09_26_drive_0106_sync"
#KITTI_OBJ_AND_IMG_DIR="/mnt/c/Users/hermann.agossou/Documents/Kitti/data/2011_09_26/2011_09_26_drive_0001_sync"
#KITTI_OBJ_AND_IMG_DIR="C:\Users\hermann.agossou\Documents\Kitti\data\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points"
KITTI_CALIB_DIR=$(realpath "./temp/data_object_calib/training/calib")
KITTI_LABEL_DIR=$(realpath "./temp/data_object_label_2/training/label_2")
KITTI_OBJ_DIR="$KITTI_OBJ_AND_IMG_DIR/velodyne_points/data"
KITTI_IMG_DIR="$KITTI_OBJ_AND_IMG_DIR/image_02/data"

# Navigate to the local directory
LOCAL_DIR_PATH="./object/2011_09_26_drive_0106_sync/"
rm "$LOCAL_DIR_PATH" -r
mkdir -p "$LOCAL_DIR_PATH"
cd "$LOCAL_DIR_PATH"


# Set number of frames
NB_FRAMES=40

# Copy and rename velodyne files
mkdir -p velodyne
cp $(find $KITTI_OBJ_DIR -type f | sort |  head -n $NB_FRAMES) velodyne -r
rename 's/0+(?=\d{6}\.bin$)//' ./velodyne/*.bin

# Copy and rename image files
mkdir -p image_2
cp $(find $KITTI_IMG_DIR -type f | sort |  head -n $NB_FRAMES) image_2 -r
rename 's/0+(?=\d{6}\.png$)//' ./image_2/*.png

# Copy calibration files
mkdir -p calib
cp $(find $KITTI_CALIB_DIR -type f | sort |  head -n $NB_FRAMES) calib -r

# Copy label files
mkdir -p label
cp $(find $KITTI_LABEL_DIR -type f | sort |  head -n $NB_FRAMES) label -r

cd $CUR_DIR
