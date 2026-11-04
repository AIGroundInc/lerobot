#!/usr/bin/env python

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

"""Test `scale_prefix_embeddings`: the sqrt(hidden_size) scaling of SmolVLA's image and language embeddings."""

import math

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, skip_if_package_missing

BATCH_SIZE = 2
NUM_LANG_TOKENS = 8


def test_prefix_scaling_defaults_to_pretrained_behavior():
    assert SmolVLAConfig().scale_prefix_embeddings


@skip_if_package_missing("transformers")
@require_cuda
def test_prefix_scaling_scales_images_and_language_only():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    set_seed(0)
    config = SmolVLAConfig(max_action_dim=7, chunk_size=4, n_action_steps=4, num_vlm_layers=2, num_steps=2)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    # Instantiate the policy on the GPU: initializing the VLM weights on the CPU is slow
    with torch.device("cuda"):
        model = SmolVLAPolicy(config).model.eval()
        images = [torch.rand(BATCH_SIZE, 3, *config.resize_imgs_with_padding) * 2 - 1]
        img_masks = [torch.ones(BATCH_SIZE, dtype=torch.bool)]
        lang_tokens = torch.randint(0, 1000, (BATCH_SIZE, NUM_LANG_TOKENS))
        lang_masks = torch.ones(BATCH_SIZE, NUM_LANG_TOKENS, dtype=torch.bool)
        state = torch.randn(BATCH_SIZE, config.max_state_dim)

    with torch.no_grad():
        scaled, pad_masks, att_masks, _ = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        config.scale_prefix_embeddings = False
        unscaled, pad_masks_2, att_masks_2, _ = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )

    torch.testing.assert_close(pad_masks_2, pad_masks)
    torch.testing.assert_close(att_masks_2, att_masks)
    factor = math.sqrt(scaled.shape[-1])
    # Image and language tokens are scaled, the state token (last) is not
    torch.testing.assert_close(
        scaled[:, :-1].float(), unscaled[:, :-1].float() * factor, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(scaled[:, -1], unscaled[:, -1])
