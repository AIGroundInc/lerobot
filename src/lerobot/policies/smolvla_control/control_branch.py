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

"""Control vision branch: camera images -> ResNet + SpatialSoftmax keypoints -> FiLM parameters."""

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from ..diffusion.modeling_diffusion import SpatialSoftmax, _replace_submodules
from .configuration_smolvla_control import SmolVLAControlConfig

# torchvision ImageNet statistics, used whatever the backbone initialization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Scale and shift for the attention input, then for the MLP input
NUM_FILM_PARAMS = 4


class KeypointEncoder(nn.Module):
    """ResNet feature map -> SpatialSoftmax keypoints, flattened to [B, 2 * num_keypoints]."""

    def __init__(
        self,
        backbone: str,
        pretrained_weights: str | None,
        use_group_norm: bool,
        image_size: tuple[int, int],
        num_keypoints: int,
    ):
        super().__init__()
        resnet = getattr(torchvision.models, backbone)(weights=pretrained_weights)
        # Drop the global average pooling and the classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        if use_group_norm:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=module.num_features // 16, num_channels=module.num_features
                ),
            )
        # Dry run in eval mode so that it does not update the BatchNorm running statistics
        self.backbone.eval()
        with torch.no_grad():
            feature_map_shape = self.backbone(torch.zeros(1, 3, *image_size)).shape[1:]
        self.backbone.train()
        self.pool = SpatialSoftmax(tuple(feature_map_shape), num_kp=num_keypoints)

    def forward(self, images: Tensor) -> Tensor:
        return self.pool(self.backbone(images)).flatten(start_dim=1)


class ControlBranch(nn.Module):
    """Encode the cameras with a trainable CNN and turn the result into FiLM parameters for every expert layer.

    Returns a [B, num_layers, 4, hidden_size] tensor holding, for each layer, the scale and shift of the
    attention input then the scale and shift of the MLP input. The last projection starts at zero, so a fresh
    branch returns zeros and leaves the expert unchanged; its weights still get gradients from the first step
    because its input (the encoded cameras) is not zero.
    """

    def __init__(self, config: SmolVLAControlConfig, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_cameras = config.num_control_cameras
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        height, width = config.control_image_size
        self.image_size = (height, width)
        self.share_encoder = config.control_share_encoder

        num_encoders = 1 if self.share_encoder else self.num_cameras
        self.encoders = nn.ModuleList(
            KeypointEncoder(
                backbone=config.control_vision_backbone,
                pretrained_weights=config.control_pretrained_backbone_weights,
                use_group_norm=config.control_use_group_norm,
                image_size=self.image_size,
                num_keypoints=config.control_num_keypoints,
            )
            for _ in range(num_encoders)
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        condition_dim = 2 * config.control_num_keypoints * self.num_cameras
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        # One linear for all layers, equivalent to one per layer
        self.film_proj = nn.Linear(hidden_size, num_layers * NUM_FILM_PARAMS * hidden_size)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def preprocess(self, image: Tensor) -> Tensor:
        """SmolVLA image in [-1, 1] (resized with padding) -> ImageNet-normalized image at `image_size`."""
        image = (image.to(self.mean.dtype) + 1.0) / 2.0
        image = F.interpolate(
            image, size=self.image_size, mode="bilinear", align_corners=False, antialias=True
        )
        return (image - self.mean) / self.std

    def encode(self, images: list[Tensor], img_masks: list[Tensor]) -> Tensor:
        """Keypoints of the real cameras, concatenated to [B, num_cameras * 2 * num_keypoints].

        `images` and `img_masks` are the outputs of `SmolVLAPolicy.prepare_images`; the trailing empty cameras
        are ignored. Cameras missing for some samples (mask False) contribute zeros.
        """
        if len(images) < self.num_cameras:
            raise ValueError(
                f"The control branch expects {self.num_cameras} cameras, got {len(images)}: set "
                "`control_num_cameras` to the number of cameras the data provides."
            )
        inputs = [self.preprocess(image) for image in images[: self.num_cameras]]
        if self.share_encoder:
            # One batched pass through the shared backbone
            keypoints = self.encoders[0](torch.cat(inputs, dim=0)).chunk(self.num_cameras, dim=0)
        else:
            keypoints = tuple(encoder(image) for encoder, image in zip(self.encoders, inputs, strict=True))
        masked = [
            camera_keypoints * mask[:, None].to(camera_keypoints.dtype)
            for camera_keypoints, mask in zip(keypoints, img_masks, strict=False)
        ]
        return torch.cat(masked, dim=-1)

    def forward(self, images: list[Tensor], img_masks: list[Tensor]) -> Tensor:
        condition = self.condition_proj(self.encode(images, img_masks))
        film = self.film_proj(condition)
        return film.view(-1, self.num_layers, NUM_FILM_PARAMS, self.hidden_size)
