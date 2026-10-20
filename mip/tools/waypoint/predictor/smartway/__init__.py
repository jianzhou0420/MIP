"""SmartWay's waypoint network and its depth encoder, vendored as-is (github.com/sxyxs/SmartWay-Code,
MIT, LICENSE here).

    waypoint_predictor/   SmartWay's ``waypoint_predictor/`` (TRM heatmap net, image-depth cross-attention,
                          nms) with the Recurrent VLN-BERT transformer it builds on (HuggingFace
                          pytorch-transformers files, Apache-2.0, headers kept)
    depth/                the DD-PPO ResNet-50 depth encoder: VLN-CE's ``resnet_encoders.py`` (MIT) and
                          habitat-lab's ``resnet.py`` / ``resnet_policy.py`` / ``running_mean_and_var.py``
                          (MIT, headers kept)

The only edits are import lines (``import utils`` -> ``from . import utils`` in TRM_net.py; the habitat
imports in depth/ -> the sibling copies), so the forward pass is upstream's.
"""
