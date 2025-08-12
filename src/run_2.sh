#!/bin/bash
exp_name="$1"
run_cmd="python main.py \
    --exp_name=$exp_name \
    --data_dir=../datasets/morphomnist \
    --hps morphomnist \
    --parents_x thickness intensity digit \
    --lr=0.0001 \
    --bs=32 \
    --wd=0.01 \
    --epochs=100 \
    --eval_freq=5 \
    --concat_pa "
if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi