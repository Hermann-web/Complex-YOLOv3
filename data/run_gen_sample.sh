#!/bin/bash

# Define the number of frames
NB_FRAMES=100

# Loop to generate the list
for ((i=0; i<NB_FRAMES; i++))
do
    printf "%06d\n" $i
done
