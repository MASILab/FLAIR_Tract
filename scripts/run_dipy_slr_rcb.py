from plumbum import local
import os
import fire

# python dipy_slr_recobundles.py
# _delete_me/double_del/Atlas_30_Bundles/whole_brain/whole_brain_MNI.trk
# _delete_me/T1_test50to250.trk "/fs5/p_masi/schwat1/spie_flair_tract_extension/resources/atlases/bundleseg/atlases/atlas/pop_average/*.trk"

parallel = local["parallel"]
fdfind = local["fdfind"]
atlases_pattern = "'/fs5/p_masi/schwat1/spie_flair_tract_extension/resources/atlases/bundleseg/atlases/atlas/pop_average/*.trk'"
wb_trk_template = "/fs5/p_masi/schwat1/spie_flair_tract_extension/resources/atlases/Atlas_30_Bundles/whole_brain/whole_brain_MNI.trk"


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
    fdfind_args = ["-t", "f", subj_pattern, search_dir, "--absolute-path"]
    input_tractogram_list = fdfind[*fdfind_args]

    total_cpus = os.cpu_count() or 1
    if jobs is None:
        jobs = max(1, total_cpus // threads_per_job)

    thread_env = {
        "OMP_NUM_THREADS": str(threads_per_job),
        "OPENBLAS_NUM_THREADS": str(threads_per_job),
        "MKL_NUM_THREADS": str(threads_per_job),
        "VECLIB_MAXIMUM_THREADS": str(threads_per_job),
        "NUMEXPR_NUM_THREADS": str(threads_per_job),
    }

    parallel_opts = ["-j", str(jobs)]
    for env_var in thread_env.keys():
        parallel_opts.extend(["--env", env_var])

    if parallel_dry_run:
        parallel_opts.append("--dry-run")
    parallel_args = parallel_opts + [
        "python",
        "dipy_slr_recobundles.py",
        wb_trk_template,
        "{}",
        atlases_pattern,
    ]

    parallel_cmd = parallel[*parallel_args]
    parallel_cmd = parallel_cmd.with_env(**thread_env)

    chain = input_tractogram_list | parallel_cmd

    if print_chain_cmd:
        print(chain)
        return
    else:
        chain()


if __name__ == "__main__":
    fire.Fire(run_dipy_slr_recobundles)
