#!/bin/bash
# properties = {properties}

# Activate virtual environment
export UV_TOOL_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/uv/tools"
export PATH="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin/:$PATH"
export UV_PYTHON_INSTALL_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/share/uv/python"
export UV_CACHE_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.cache"
export UV_PYTHON_BIN_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin"
export UV_TOOL_BIN_DIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/bin"
export IPYTHONDIR="/fs5/p_masi/schwat1/venv_configs_for_fs5/.local/.ipython"

source /fs5/p_masi/schwat1/spie_flair_tract_extension/.venv/bin/activate

# Execute the rule commands
{exec_job}
