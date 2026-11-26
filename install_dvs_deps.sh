set -e
PY=/e/ProgramData/miniconda3/envs/dvs/python.exe
MIRROR=https://mirror.nju.edu.cn/pypi/web/simple

echo "=== 1) torch + torchvision (PyPI 上的 Windows wheel 自带 CUDA 12.4 运行时) ==="
$PY -m pip install -i $MIRROR \
    torch==2.5.1 torchvision==0.20.1
echo "=== 2) 其余依赖 ==="
$PY -m pip install -i $MIRROR \
    numba pycocotools tqdm scipy matplotlib filterpy lap omegaconf hydra-core tensorboard
echo "=== 3) 校验 ==="
$PY -c "import torch, torchvision; print('torch', torch.__version__, '| cuda runtime', torch.version.cuda, '| gpu', torch.cuda.is_available()); print('torchvision', torchvision.__version__)"
$PY -c "import h5py, numpy, yaml, cv2, numba, pycocotools, tqdm, scipy; print('其余依赖 OK')"
