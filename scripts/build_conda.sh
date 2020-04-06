#!/bin/bash
# File   : database.py
# Author : Raeid Saqur
# Email  : raeidsaqur@cs.toronto.edu
# Date   : 09/23/2019
#
# This file is part of PGFM Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

pkg='clevr-parser'
CONDA_PATH="${HOME}/anaconda3"
CONDA_RECIPE_PATH="${HOME}/${pkg}"
echo -e "\tCONDA_PATH: ${CONDA_PATH}"
#py_vers=( 3.7 3.8 )
py_vers=( 3.7 )
#platforms=( osx-64 linux-32 linux-64 win-32 win-64 )
platforms=( osx-64 linux-64 )
#platforms=( all )

echo "Building conda package ..."
cd ~
echo "Creating conda recipes in: ${PWD}"
conda skeleton pypi ${pkg}    # Creates the CONDA_RECIPE_PATH

#cd ${pkg}
#wget https://conda.io/docs/_downloads/build1.sh
#wget https://conda.io/docs/_downloads/bld.bat
#cd ~

for v in "${py_vers[@]}"; do
  echo "conda-build --python ${v} ${pkg}"
  conda-build --python $v $pkg
done

find ${CONDA_PATH}/conda-bld/linux-64/ -name *.tar.bz2 | while read file; do
  echo "Converting file: ${file} ..."
  for platform in "${platforms[@]}"; do
    conda convert --platform $platform $file -o ${CONDA_PATH}/conda-bld/
  done
done

## Upload to Anaconda Cloud ##
# Pre-req: $ conda install anaconda-client; $ anaconda login ; $ anaconda whoami
echo "Uploading to Anaconda Cloud [Presuming anaconda-client is installed and logged in]"
find ${CONDA_PATH}/conda-bld/ -name *.tar.bz2 | while read file; do
    echo "Uploading file: $file ..."
    anaconda upload $file
done
wait;

echo "Building and Uploading Completed on:"
date;

## Clean-up ##
echo "Clean-up: Removing ${CONDA_RECIPE_PATH}"
rm -rf ${CONDA_RECIPE_PATH}
#Purge src and build intermediaries #
echo "Clean-up: conda-build purge"
conda-build purge
wait;
echo "Clean-up done"

exit 0







