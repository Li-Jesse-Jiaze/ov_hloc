#!/usr/bin/env bash

# Source our workspace directory to load ENV variables
source /home/patrick/workspace/catkin_ws_ov/devel/setup.bash


#=============================================================
#=============================================================
#=============================================================


# estimator configurations
modes=(
    "mono"
    "stereo"
)

# dataset locations
bagnames=(
    "dataset-room1_512_16"
    "dataset-room2_512_16"
    "dataset-room3_512_16"
    "dataset-room4_512_16"
    "dataset-room5_512_16"
    "dataset-room6_512_16"
)



#=============================================================
#=============================================================
#=============================================================


# Loop through all modes
for h in "${!modes[@]}"; do
# Loop through all datasets
for i in "${!bagnames[@]}"; do

# Monte Carlo runs for this dataset
# If you want more runs, change the below loop
for j in {00..04}; do

# start timing
start_time="$(date -u +%s)"

# number of cameras
if [ "${modes[h]}" == "mono" ]
then
    temp1="1"
fi
if [ "${modes[h]}" == "stereo" ]
then
    temp1="2"
fi

# run our ROS launch file (note we send console output to terminator)
roslaunch loop_fusion record_tumvi.launch max_cameras:="$temp1" dataset:="${bagnames[i]}" run_number:="$j" mode_type:="${modes[h]}" &> /dev/null

# print out the time elapsed
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "BASH: ${modes[h]} - ${bagnames[i]} - run $j took $elapsed seconds";

done


done
done


