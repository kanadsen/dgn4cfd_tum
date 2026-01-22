
#PBS -1 walltime=1:00:00
#PBS -q debug
#PBS -1 filesystems=grand:home
#PBS -A NeuralDE

PROJECT_DIR=/lus/grand/projects/NeuralDE/kanadsen/myrepos/dgn4cfd_tum/examples/ARO
cd PROJECT_DIR || exit 1

source ../../../../environment_folders/dgn_new_env/bin/activate

python train_dgn.py