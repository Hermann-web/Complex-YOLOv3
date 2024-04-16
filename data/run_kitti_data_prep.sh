#!/bin/bash
# Script: kitti_data_prep.sh
# Description: Prepare KITTI dataset for object detection
# Author: Hermann Agossou hermannagossou7[at]gmail[dot]com
# Date: 2024/03/29

# Check if enum argument is provided

USAGE="Usage: [--split split] [--use-subfolder] [--keep-folder] [--max-frames]"

if [ $# -lt 1 ]; then
    echo $USAGE
    return 1
fi

DATA_LABEL="2011_09_26_drive_0005_sync"
KITTI_OBJ_AND_IMG_DIR="./temp/2011_09_26/2011_09_26_drive_0005_sync"
KITTI_CALIB_DIR="./temp/data_object_calib/training/calib"
KITTI_LABEL_DIR="./temp/data_object_label_2/training/label_2"


USE_SPLIT_SET_FILE=false
SPLIT_LABEL="all"
USE_SUBFOLDER=false
USE_NB_FRAMES=false
MAX_FRAMES=0
KEEP_EXISTING_FOLDER=false

for (( i=1; i<=$#; i++ )); do
    if [[ "${!i}" == "--split" ]]; then
        # Check if split is specified
        next_arg_index=$((i + 1))
        if [ "$next_arg_index" -le "$#" ]; then
            SPLIT_LABEL="${!next_arg_index}"
            USE_SPLIT_SET_FILE=true
        fi
    fi
    if [[ "${!i}" == "--use-subfolder" ]]; then
        # Check if flag is specified
        USE_SUBFOLDER=true
    fi
    if [[ "${!i}" == "--max-frames" ]]; then
        # Check if split is specified
        next_arg_index=$((i + 1))
        if [ "$next_arg_index" -le "$#" ]; then
            MAX_FRAMES=${!next_arg_index}
            USE_NB_FRAMES=true
        fi
    fi
    if [[ "${!i}" == "--keep-folder" ]]; then
        # Check if purge is specified
        KEEP_EXISTING_FOLDER=true
    fi
    if [[ "${!i}" == "--help" ]]; then
        echo $USAGE
        return 1
    fi
done


echo "USE_SPLIT_SET_FILE: $USE_SPLIT_SET_FILE"
echo "SPLIT_LABEL: $SPLIT_LABEL"
echo "USE_SUBFOLDER: $USE_SUBFOLDER"
echo "USE_NB_FRAMES: $USE_NB_FRAMES"
echo "MAX_FRAMES: $MAX_FRAMES"
echo "KEEP_EXISTING_FOLDER: $KEEP_EXISTING_FOLDER"

# Function to set subfolder_name and image_set_file based on enum argument
set_subfoler_and_split_file() {
    if [ "$1" = "train" ]; then
        subfolder_name="training"
        image_set_file=./ImageSets/train.txt
    elif [ "$1" = "test" ]; then
        subfolder_name="testing"
        image_set_file=./ImageSets/test.txt
    elif [ "$1" = "valid" ]; then
        subfolder_name="training"
        image_set_file=./ImageSets/valid.txt
    elif [ "$1" = "sample2" ]; then
        subfolder_name="sample2"
        image_set_file=./ImageSets/sample2.txt
    elif [ "$1" = "all" ]; then
        subfolder_name="all"
        image_set_file=""
    else
        echo "Invalid argument. Usage: $0 [train|test|valid|sample2|all]"
        return 1
    fi
    return 0
}

# Call set_subfoler_and_split_file function with the provided argument
set_subfoler_and_split_file "$SPLIT_LABEL" || return 1

# ----------------resolve absolute paths--------------------

CUR_DIR="$PWD"
cd data/KITTI

# if using image_set_file, set realpath, relative to data/kitti
if [ "$USE_SPLIT_SET_FILE" = true ]; then
    image_set_file=$(realpath $image_set_file)
fi

KITTI_OBJ_AND_IMG_DIR=$(realpath "$KITTI_OBJ_AND_IMG_DIR")
KITTI_CALIB_DIR=$(realpath "$KITTI_CALIB_DIR")
KITTI_LABEL_DIR=$(realpath "$KITTI_LABEL_DIR")
KITTI_OBJ_DIR="$KITTI_OBJ_AND_IMG_DIR/velodyne_points/data"
KITTI_IMG_DIR="$KITTI_OBJ_AND_IMG_DIR/image_02/data"

if [ "$USE_SUBFOLDER" = true ]; then
    LOCAL_DIR_PATH="./object/$subfolder_name/"
else
    LOCAL_DIR_PATH="./object/$DATA_LABEL/"
fi
LOCAL_DIR_PATH_ABS=$(realpath "$LOCAL_DIR_PATH")

cd $CUR_DIR


# -----------------resolve the selection settings-------------------

mkdir -p "$LOCAL_DIR_PATH_ABS"

if [ "$KEEP_EXISTING_FOLDER" = false ]; then
    rm "$LOCAL_DIR_PATH_ABS" -r
    mkdir -p "$LOCAL_DIR_PATH_ABS"
fi

# if using image_set_file
if [ "$USE_SPLIT_SET_FILE" = true ]; then
    img_ids=$(cat $image_set_file)
else
    # if not using one
    img_ids=$(find $KITTI_LABEL_DIR -type f -name "{}.txt")    
fi

# if using number of frames
if [ "$USE_NB_FRAMES" = true ]; then
    NB_FRAMES=$MAX_FRAMES
else
    # if not using one
    NB_FRAMES=$(echo $img_ids | wc -l)
fi

# wrapping up
img_ids=$(echo $img_ids | head -n $NB_FRAMES)

echo "nb of images =$(echo $img_ids | tr " " "\n" | wc -l )"

# ----------------handle the selection--------------------

cd "$LOCAL_DIR_PATH_ABS"

# Copy and rename velodyne files
echo "copying velodyne files to $(realpath .)/velodyne"
mkdir -p velodyne
echo $img_ids | tr " " "\n" | xargs -I{} find $KITTI_OBJ_DIR -type f -name "0000{}.bin" | xargs -I{} sh -c 'cp {} "velodyne/$(basename {} | sed "s/^0000//")"'

# Copy and rename image files
echo "copying image2 files to $(realpath .)/image_2"
mkdir -p image_2
echo $img_ids | tr " " "\n" | xargs -I{} find $KITTI_IMG_DIR -type f -name "0000{}.png" | xargs -I{} sh -c 'cp {} "image_2/$(basename {} | sed "s/^0000//")"'

# Copy calibration files
echo "copying calibration files to $(realpath .)/calib"
mkdir -p calib
echo $img_ids | tr " " "\n" | xargs -I{} find $KITTI_CALIB_DIR -type f -name "{}.txt" | xargs -I{} cp {} calib

# Copy label files
echo "copying label files to $(realpath .)/label_2"
mkdir -p label
echo $img_ids | tr " " "\n" | xargs -I{} find $KITTI_LABEL_DIR -type f -name "{}.txt" | xargs -I{} cp {} label_2

cd $CUR_DIR
echo "done."

echo "SPLIT_LABEL=$SPLIT_LABEL"
echo "FOLDER=$LOCAL_DIR_PATH"
