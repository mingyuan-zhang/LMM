export PYTHONPATH=../../../../:$PYTHONPATH
# Get segments from annotation
echo "Process segments"
unzip -q pose_data/humanact12.zip -d pose_data/
python process_segment.py

# Generate motion data, meta data, text data and text features
echo "Create dataset"
unzip -q texts.zip
python create_dataset.py

# Generate official motion data for evaluation
echo "Generate official motion data"
python process_official_segment.py
python process_eval_motion.py
cp -r new_joint_vecs ../../datasets/humanml3d/eval_motions