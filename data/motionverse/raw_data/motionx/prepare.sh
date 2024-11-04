export PYTHONPATH=../../../../:$PYTHONPATH
# Unzip and rename files
# echo "Unzip and rename files"
# jar xf motionx_face_motion_data.zip
# jar xf motionx_seq_text_face.zip
# jar xf motionx_seq_text.zip
# jar xf motionx_smplx.zip

# Generate motion data, meta data, text data and text features
echo "Create dataset"
python create_dataset.py