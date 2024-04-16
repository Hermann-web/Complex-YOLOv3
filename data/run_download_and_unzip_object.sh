#!/bin/bash
# original code: https://github.com/hermann-web/Open3D-PointNet2-Semantic3D

CUR_DIR=$(realpath "$PWD")
cd data/KITTI

# Create directory if it doesn't exist
mkdir -p temp

cd temp

files=(
    2011_09_26_calib.zip
    2011_09_26_drive_0005
)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done

cd $CUR_DIR
