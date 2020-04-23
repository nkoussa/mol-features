#!/bin/bash
# Summit
# bsub -Is -W 0:25 -nnodes 1 -P MED106 $SHELL
# --------------------------------------------------------
# Theta
# qsub -I -t 30 -n 1 -A CVD_Research -q debug-flat-quad
# www.alcf.anl.gov/support-center/theta/job-scheduling-policy-theta#queues
# --------------------------------------------------------

# dump_prefix="/gpfs/alpine/med106/scratch/$USER"
dump_prefix="./"
  
i1=$1
i2=$2
par_jobs=$3
gout="$dump_prefix"
echo "Run smiles ids: $i1-$i2"
echo "Global output: $gout"

# aprun -n 4 -N 4 ./aprun_script.sh 100 120 4 > run.log
python src/BL2/gen_mordred_1.py --i1 $i1 --i2 $i2 --par_jobs $par_jobs --gout $gout > "$dump_prefix"/ids."$i1"-"$i2".log 2>&1

