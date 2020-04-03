#!/bin/bash 

mkdir -p data/raw
cd data/raw

echo "Downloading CLEVR dataset..."
#wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
rm CLEVR_v1.0.zip

cd ..  # root