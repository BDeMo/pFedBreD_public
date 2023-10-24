#!/bin/sh

gpu_t="0"
dataset_t="sent140"
name_t="pFedBreD_ns_fo"
model_t="lstm"
loss_name_t="NLLLoss"
learning_rate_t="1e-2"
num_aggregate_locals_t="4"
batch_size_t="400"
beta_t="1.0"
#beta_t="2.0"
local_epochs_t="20"
personal_learning_rate_t="1e-2"
lamda_t="15.0"
prox_iters_t="5"
total_epochs_t="800"
optimizer_name_t="SGD"
times_t="5"
eta_t="5e-2"

for name_t in FedAvg PerFedAvg pFedMe pFedBreD_ns_mh
  do
    python main_fl.py --gpu $gpu_t --dataset $dataset_t --name $name_t --model $model_t --loss_name $loss_name_t --learning_rate $learning_rate_t --num_aggregate_locals $num_aggregate_locals_t --batch_size $batch_size_t --beta $beta_t   --local_epochs $local_epochs_t --personal_learning_rate $personal_learning_rate_t --lamda $lamda_t --prox_iters $prox_iters_t --eta $eta_t --total_epochs $total_epochs_t --optimizer_name $optimizer_name_t --times $times_t
  done
