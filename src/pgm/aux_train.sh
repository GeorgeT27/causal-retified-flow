#!/bin/bash
model_name='aux_60k'
exp_name=$model_name
parents='t_i_d'
mkdir -p "../../checkpoints/$parents/$exp_name"


run_cmd="python train_pgm.py \
    --exp_name=$exp_name \
    --dataset morphomnist \
    --data_dir=/workspace/causal-retified-flow/datasets/morphomnist
     \
    --hps morphomnist \
    --setup sup_aux \
    --parents_x ["thickness", "intensity", "digit"] \
    --concat_pa False \
    --lr=0.001 \
    --bs=32 \ "

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
