#!/usr/bin/env bash


export UV_TOOL_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/uv/tools"
export PATH="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin/:$PATH"
export UV_PYTHON_INSTALL_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/share/uv/python"
export UV_CACHE_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.cache"
export UV_PYTHON_BIN_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin"
export UV_TOOL_BIN_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin"
export IPYTHONDIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/.ipython"


export pybin=/fs5/p_masi/schwat1/spie_flair_tract_extension/.venv/bin/python
export generate_script=/fs5/p_masi/schwat1/spie_flair_tract_extension/model/generate.py
data_dir=/valiant02/masi/schwat1/projects/spie_flair_tract_extension/FLAIR_processing/BIDS_format/derivatives
job_log="$data_dir/inference_job.log"

fdfind fod_mni_trix "$data_dir" --absolute-path | while read -r file_path; do
  dir_path=$(dirname "$file_path")
  if [ -e "$dir_path/T1_atlas_registered" ]; then
    echo "$dir_path"
  fi
done > "$data_dir/subj_for_inference.txt"


run_ft_inference() {
  local slot="$1"
  local sub_dir="$2"
  
  if [ -z "$slot" ] || [ -z "$sub_dir" ]; then
    echo "Error: Missing slot or sub_dir argument for run_ft_inference." >&2
    return 1
  fi

  slot=$((slot - 1))
  export CUDA_VISIBLE_DEVICES="$slot"
  "$pybin" "$generate_script" "$sub_dir"
}

export -f run_ft_inference

parallel --env _ --progress --joblog "$job_log" -j3 run_ft_inference {%} {}  :::: "$data_dir"/subj_for_inference.txt
