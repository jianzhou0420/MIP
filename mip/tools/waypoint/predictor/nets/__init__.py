"""The predictor's three networks, cut down to their inference paths from the upstream code (attribute
names kept, so the released checkpoints load by key; dropout and training-only parts left out):

    dinov2.py     DINOv2 ViT-S/14-reg     facebookresearch/dinov2            Apache-2.0   LICENSE-dinov2
    depth.py      DD-PPO ResNet-50        facebookresearch/habitat-lab, VLN-CE   MIT     LICENSE-habitat-lab, LICENSE-vlnce
    waypoint.py   SmartWay's TRM + NMS    sxyxs/SmartWay-Code                MIT          LICENSE-smartway
                  its BERT layers         huggingface pytorch-transformers   Apache-2.0   (LICENSE-dinov2 is the same text)
"""
