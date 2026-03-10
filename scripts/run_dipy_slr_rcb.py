from plumbum import local
import os
import fire
import tempfile
import yaml
from more_itertools import flatten

parallel = local["parallel"]
fdfind = local["fdfind"]

with open("rdsr.yaml", "r") as config:
    conf = yaml.safe_load(config)

atlases_pattern = conf["atlases_pattern"]
wb_trk_template = conf["wb_trk_template"]
script_path = conf["script_path"]


def run_dipy_slr_recobundles(
    search_dir,
    parallel_dry_run=False,
    print_chain_cmd=False,
    atlases_pattern=atlases_pattern,
    wb_trk_template=wb_trk_template,
    subj_pattern="T1_test50to250.trk",
    threads_per_job=4,
    jobs=None,
):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        input_list_path = tmp.name
    fdfind_args = ["-t", "f", subj_pattern, search_dir, "--absolute-path"]
    (fdfind[fdfind_args] > input_list_path).run()

    total_cpus = os.cpu_count() or 1
    if jobs is None:
        jobs = max(1, total_cpus // threads_per_job)

    thread_env = {
        "OMP_NUM_THREADS": str(threads_per_job),
        "OPENBLAS_NUM_THREADS": str(threads_per_job),
        "MKL_NUM_THREADS": str(threads_per_job),
        "VECLIB_MAXIMUM_THREADS": str(threads_per_job),
        "NUMEXPR_NUM_THREADS": str(threads_per_job),
    } | conf["env_vars"]

    parallel_opts = ["-j", str(jobs)]
    for env_var in thread_env.keys():
        parallel_opts.extend(["--env", env_var])

    if parallel_dry_run:
        parallel_opts.append("--dry-run")

    remotes = list(flatten([("-S", remote) for remote in conf["remotes"]]))
    parallel_args = parallel_opts + remotes + [
        "--bf",
        input_list_path,
        "-a",
        input_list_path,
        conf["env_vars"]["pybin"],
        conf["script_path"],
        wb_trk_template,
        "{}",
        atlases_pattern,
    ]

    parallel_cmd = parallel[parallel_args]
    parallel_cmd = parallel_cmd.with_env(**thread_env)


    if print_chain_cmd:
        print(parallel_cmd)
        return
    else:
        # print(parallel_cmd)
        out = parallel_cmd.run()
        print(out)


if __name__ == "__main__":
    fire.Fire(run_dipy_slr_recobundles)
