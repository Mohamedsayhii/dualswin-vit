sudo apt update && sudo apt install rsync 

git clone -b cloud-training --single-branch https://github.com/Mohamedsayhii/dualswin-vit.git

curl -L -o ade20k.zip https://www.kaggle.com/api/v1/datasets/download/awsaf49/ade20k-dataset

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n dualswin python=3.9 -y
conda activate dualswin

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

pip install mmengine==0.10.7 mmdet==2.28.2 mmsegmentation==0.30.0 numpy==1.26.4 timm future tensorboard pywavelets

python - <<'EOF'
import torch, mmcv, mmengine, mmdet, mmseg
print("torch:", torch.__version__)
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("mmdet:", mmdet.__version__)
print("mmseg:", mmseg.__version__)
EOF
