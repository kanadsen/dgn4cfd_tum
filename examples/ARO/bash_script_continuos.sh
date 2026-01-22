#!/bin/bash
#Shell script that submits ml4dvar_pangu jobs to allow for continual cycling using
#Watches if current job is completed and then cycles the job
START=1
NUM_JOBS=32
for i in $(seq $START $NUM_JOBS)
    do
        echo "Starting $i"
        echo $i
        qsub -N da_job"$i" -W block=true -v da_type=$2,exp_name=$3 $1
done