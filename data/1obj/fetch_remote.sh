#!/bin/bash

REMOTE_DATA_DIR='/n/fs/pnlp/rsaqur/projects/compositionality/clevr-dataset-gen/output/oct8'
##- Fetch Images
#item=images
#split=val
#fn='CLEVR_'${split}'_00000*.png'
#REMOTE_DATA_DIR='/n/fs/pnlp/rsaqur/projects/compositionality/clevr-dataset-gen/output/oct8'
#scp rsaqur@ionic.cs.princeton.edu:${REMOTE_DATA_DIR}/${item}/${split}/${fn} ./${item}/${split}/
#

# Fetch Scene
item=scenes
for split in 'test' 'train' 'val'; do
  fn='CLEVR_'${split}'_00000*.json'
  scp rsaqur@ionic.cs.princeton.edu:${REMOTE_DATA_DIR}/${item}/${split}/${fn} ./${item}/${split}/
done;

## Fetch Question
#item=questions
#split=train
#fn='CLEVR_train_000001.json'
#REMOTE_DATA_DIR='/n/fs/pnlp/rsaqur/projects/compositionality/clevr-dataset-gen/output/oct8'
#scp rsaqur@ionic.cs.princeton.edu:${REMOTE_DATA_DIR}/${item}/* ./${item}/