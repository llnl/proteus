set -e

ml load python/3.9

echo "DIRS are under"
python -m venv ${SYS_TYPE}-${PROTEUS_CI_ROCM_VERSION}
source ${SYS_TYPE}-${PROTEUS_CI_ROCM_VERSION}/bin/activate
LLVM_INSTALL_DIR=${ROCM_PATH} PROTEUS_SRC=${CI_PROJECT_DIR} pip install -v git+https://github.com/LLNL/Mneme.git@proteus-ci
