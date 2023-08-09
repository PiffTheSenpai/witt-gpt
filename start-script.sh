#!/bin/bash
cd $SCRATHDIR
git clone witt-train
cd witt-train
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:21.05-py3.SIF 
