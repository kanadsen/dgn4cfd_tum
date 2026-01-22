#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle:grand
#PBS -A NeuralDE
#PBS -m bae

set -x
set -e

PROJECT_DIR=/lus/grand/projects/NeuralDE/kanadsen/myrepos/dgn4cfd_tum/examples/ARO
ls -ld $PROJECT_DIR
cd $PROJECT_DIR || exit 1

echo "Project Dir changed"

source /lus/grand/projects/NeuralDE/kanadsen/environment_folders/dgn_new_env/bin/activate

echo "Source activated"
which python
python -u train_dgn.py 