from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/RPI/poly_oscillatorconti_221116-234246"] * 2,
    trained_policy_iteration_list=["40", "50"],
    is_init_info=True,
    init_info={"init_state": [1, -1]},
    save_render=False,
    legend_list=["RPI-40", "RPI-50"],
    use_opt=True,
    constrained_env=False,
    is_tracking=False,
    opt_args={"opt_controller_type": "OPT"},
    dt=None,
)

runner.run()