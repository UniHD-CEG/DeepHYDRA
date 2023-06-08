#!/usr/bin/env bash

conda activate informers
source /cvmfs/atlas.cern.ch/repo/sw/tdaq/tools/cmake_tdaq/bin/cm_setup.sh -r tdaq-10-00-00 x86_64-centos7-gcc11-opt
export PYTHONPATH=/eos/user/k/kstehle/anaconda/envs/informers/lib/python3.9/site-packages:$PYTHONPATH
export PBEAST_SERVER_SSO_SETUP_TYPE=AutoUpdateKerberos
