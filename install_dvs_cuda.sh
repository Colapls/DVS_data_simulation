set -e
PY=/e/ProgramData/miniconda3/envs/dvs/python.exe
# 南大 pytorch 镜像（国内可用的 cu124 wheel 源；清华/阿里对本仓库 403）
$PY -m pip install -i https://mirror.nju.edu.cn/pytorch/whl/cu124 \
    --extra-index-url https://mirror.nju.edu.cn/pypi/web/simple \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124
$PY -c "import torch, torchvision; print('torch', torch.__version__, '| cuda runtime', torch.version.cuda, '| gpu', torch.cuda.is_available()); print('torchvision', torchvision.__version__)"
