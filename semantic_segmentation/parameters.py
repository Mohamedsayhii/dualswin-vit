import torch
from mmseg.models import build_segmentor
from mmcv import Config

cfg = Config.fromfile('configs/dualvit/upernet_dualvit_s_512x512_160k_ade20k.py')

model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

model = model.cuda()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
backbone_params = sum(p.numel() for p in model.backbone.parameters())

print(f"Backbone parameters: {backbone_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

