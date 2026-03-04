#!/usr/bin/env bash


export UV_TOOL_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/uv/tools"
export PATH="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin/:$PATH"
export UV_PYTHON_INSTALL_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/share/uv/python"
export UV_CACHE_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.cache"
export UV_PYTHON_BIN_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin"
export UV_TOOL_BIN_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin"
export IPYTHONDIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/.ipython"


export pybin=/fs5/p_masi/schwat1/spie_flair_tract_extension/.venv/bin/python
export process_script=/fs5/p_masi/schwat1/spie_flair_tract_extension/preprocessing/process_flair.py
data_dir=/valiant02/masi/schwat1/projects/spie_flair_tract_extension/FLAIR_processing/BIDS_format
job_log="$data_dir/flair_preproc_job.log"

fdfind -t 'd' "^anat" "$data_dir" --absolute-path > "$data_dir/flair_preproc.txt"


run_flair_proc() {
  local sub_dir="$1"

  "$pybin" "$process_script" "$sub_dir"
}

export -f run_flair_proc

parallel -S 16/masi-74.vuds.vanderbilt.edu -S 32/masi-celgate2.vuds.vanderbilt.edu --env _ --progress --joblog "$job_log" --resume -P1 run_flair_proc {//}  :::: "$data_dir/flair_preproc.txt"
