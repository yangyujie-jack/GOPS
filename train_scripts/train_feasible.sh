python ../example_train/fhadp_feasible/fhadp_feasible_mlp_veh3dof_tracking_detour_serial.py \
    --pre_horizon 10 --eval_accuracy True \
& sleep 5
python ../example_train/fhadp_feasible/fhadp_feasible_mlp_veh3dof_tracking_detour_serial.py \
    --pre_horizon 20 --eval_accuracy True \
& sleep 5
python ../example_train/fhadp_feasible/fhadp_feasible_mlp_veh3dof_tracking_detour_serial.py \
    --pre_horizon 30 --eval_accuracy True \
& sleep 5
python ../example_train/fhadp_feasible/fhadp_feasible_mlp_veh3dof_tracking_detour_serial.py \
    --pre_horizon 40 --eval_accuracy True \
