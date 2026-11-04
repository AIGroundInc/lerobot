# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_control")
@dataclass
class SmolVLAControlConfig(SmolVLAConfig):
    """SmolVLA with a trainable control vision branch that FiLM-modulates the action expert.

    Every real camera goes through a ResNet + SpatialSoftmax keypoint encoder (as in Diffusion Policy). The
    keypoints of all cameras are projected to one condition vector, which scales and shifts the normalized
    hidden states of the action expert before the attention and before the MLP of every expert layer. The
    VLM conditioning (KV cache, cross-attention) is unchanged. The modulation starts at zero, so a fresh
    branch reproduces the base SmolVLA policy exactly and can be added to existing SmolVLA checkpoints.
    """

    # torchvision ResNet used by the control branch, cut before its global pooling.
    control_vision_backbone: str = "resnet18"
    # torchvision weights of the backbone; None trains it from a random initialization.
    control_pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    # Replace BatchNorm by GroupNorm in the backbone (only without pretrained weights).
    control_use_group_norm: bool = False
    # (H, W) the SmolVLA-preprocessed images are resized to before the backbone.
    control_image_size: tuple[int, int] = (256, 256)
    # SpatialSoftmax keypoints per camera; each contributes its (x, y) to the condition.
    control_num_keypoints: int = 32
    # One backbone for all cameras instead of one per camera.
    control_share_encoder: bool = True
    # Cameras fed to the control branch: the first ones of `image_features`; None uses every real camera. Set it
    # when the checkpoint declares cameras the data does not have (e.g. `smolvla_base` declares 3 cameras and
    # LIBERO has 2): `prepare_images` drops the missing cameras, so the branch would get fewer images.
    control_num_cameras: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if not self.control_vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`control_vision_backbone` must be a torchvision ResNet, got {self.control_vision_backbone}."
            )
        if self.control_use_group_norm and self.control_pretrained_backbone_weights is not None:
            raise ValueError(
                "`control_use_group_norm` replaces the BatchNorm layers, which would ruin the pretrained "
                "backbone weights: set `control_pretrained_backbone_weights=null` to use it."
            )
        if len(self.control_image_size) != 2 or any(size <= 0 for size in self.control_image_size):
            raise ValueError(
                f"`control_image_size` must be a pair of positive integers, got {self.control_image_size}."
            )
        if self.control_num_keypoints < 1:
            raise ValueError(f"`control_num_keypoints` must be >= 1, got {self.control_num_keypoints}.")
        if self.control_num_cameras is not None and self.control_num_cameras < 1:
            raise ValueError(f"`control_num_cameras` must be >= 1, got {self.control_num_cameras}.")

    def validate_features(self) -> None:
        super().validate_features()
        if self.num_real_cameras < 1:
            raise ValueError("The control branch needs at least one image feature besides the empty cameras.")
        if self.control_num_cameras is not None and self.control_num_cameras > self.num_real_cameras:
            raise ValueError(
                f"`control_num_cameras={self.control_num_cameras}` exceeds the {self.num_real_cameras} image "
                "features besides the empty cameras."
            )

    @property
    def num_real_cameras(self) -> int:
        """Image features besides the empty cameras, which `prepare_images` appends (always masked) last."""
        return len(self.image_features) - self.empty_cameras

    @property
    def num_control_cameras(self) -> int:
        """Cameras fed to the control branch."""
        return self.control_num_cameras if self.control_num_cameras is not None else self.num_real_cameras
