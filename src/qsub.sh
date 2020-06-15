#!/bin/bash
#COBALD -t 0:05
#COBALD -n 1
#COBALD -q default
#COBALD -A CVD_Research 

# (ap) Note! I didn't think have run this script.
echo "Starting Cobalt job script"

SAMPLES=976019
NODES=16
CORES=16
SAMPLES_PER_NODE=$(( $SAMPLES / $NODES ))

# SEQ=$(seq 0 $SAMPLES_PER_NODE $SAMPLES) 
SEQ=$(seq -1 $SAMPLES_PER_NODE $SAMPLES) 
echo "Number of smiles to process: $SAMPLES"
echo "Number of nodes: $NODES"
echo "Number of samples per node: $SAMPLES_PER_NODE"
echo $SEQ

for ii in $SEQ; do
    i1=$(($ii + 1))
    i2=$(($ii + $SAMPLES_PER_NODE))
    echo "Process indices: $i1, $i2"
    # aprun -n 1 -N 4 ./aprun_script.sh $i1 $i2 $CORES exec > ids."$i1"-$i2"".log 2>&1 
    # aprun -n 1 -a 1 -c 4 -g 1 ./aprun_script.sh $i1 $i2 $CORES exec >run0.log 2>&1
done


