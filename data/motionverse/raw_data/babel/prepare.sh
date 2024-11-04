export PYTHONPATH=../../../../:$PYTHONPATH
# Unzip and rename files
echo "Unzip and rename files"
unzip babel_v1-0_release.zip

# Generate motion data, meta data, text data and text features
echo "Create dataset"
python create_dataset.py