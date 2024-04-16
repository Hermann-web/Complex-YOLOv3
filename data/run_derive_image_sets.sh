#!/bin/bash

SET_LABEL=train
SET_DIR_LABEL=training

SET_LABEL=test
SET_DIR_LABEL=testing


KITTI_OBJ_DIR=KITTI/object/$SET_DIR_LABEL/velodyne
TRAIN_SET=KITTI/ImageSets/$SET_LABEL.txt
TRAIN_SET2=KITTI/ImageSets2/$SET_LABEL.txt

mkdir -p KITTI/ImageSets2

find $KITTI_OBJ_DIR -type f -name "*.bin" -exec grep $TRAIN_SET

cat $TRAIN_SET | xargs -I{} find $KITTI_OBJ_DIR -type f -name "{}.bin" > $TRAIN_SET2

echo $FILES
