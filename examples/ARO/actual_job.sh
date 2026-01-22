
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=grand:home
#PBS -A NeuralDE

set -x
set -e

PROJECT_DIR=/lus/grand/projects/NeuralDE/kanadsen/myrepos/dgn4cfd_tum/examples/ARO
ls -ld $PROJECT_DIR
cd $PROJECT_DIR || exit 1

source /lus/grand/projects/NeuralDE/kanadsen/environment_folders/dgn_new_env/bin/activate

echo "Source activated"
which python
python -u train_dgn.py