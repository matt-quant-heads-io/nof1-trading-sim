import copy
import hydra
import submitit

from illuminate import get_arg_parser, main as main_qd


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    seeds = list(range(5))
    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
        slurm_job_name=f"crypto_qd",
        mem_gb=30,
        tasks_per_node=1,
        cpus_per_task=30,
        timeout_min=1440,
        slurm_account='pr_174_tandon_advanced', 
    )
    sweep_configs = []
    for seed in seeds:
        cfg = copy.deepcopy(args)
        cfg.seed = seed
        cfg.user_ray = True
        cfg.num_cpus = 30
        cfg.non_random_start = True
        sweep_configs.append(cfg)

    executor.map_array(
        main_qd,
        sweep_configs,
    )


if __name__ == "__main__":
    main()