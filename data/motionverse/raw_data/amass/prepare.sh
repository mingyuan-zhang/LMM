#!/bin/bash

# unzip all files
echo "Unzip AMASS data files"
data_path="amass_data"

for file in `ls $data_path`
do
    if [ "${file##*.}"x = "bz2"x ]; then
        echo $file
        tar -jxf "$data_path/$file" -C $data_path
    fi
done

export PYTHONPATH=../../../../:$PYTHONPATH
# process pose data
echo "Process pose data"
python process_pose.py

echo "Creat dataset"
python create_dataset.py