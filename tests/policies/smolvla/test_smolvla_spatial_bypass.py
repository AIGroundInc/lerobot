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

"""Test the SmolVLA spatial bypass (gated cross-attention from the expert to pooled SigLIP patches)."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SpatialBypass
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, skip_if_package_missing

BATCH_SIZE = 2
CHUNK_SIZE = 10
NUM_CAMERAS = 2


def test_spatial_bypass_pools_patches_in_grid_order():
    bypass = SpatialBypass(
        vision_hidden_size=1, num_key_value_heads=1, head_dim=2, gated_layers=[1, 3], num_cameras=1
    )
    # 4x4 grid flattened row-major, the value of each patch is its index
    patches = torch.arange(16, dtype=torch.float32).view(1, 16, 1)

    pooled = bypass.embed_patches(patches, camera_idx=0)

    assert pooled.shape == (1, 4, 1)
    torch.testing.assert_close(pooled.flatten(), torch.tensor([2.5, 4.5, 10.5, 12.5]))


def test_spatial_bypass_kv_shapes_and_gates():
    bypass = SpatialBypass(
        vision_hidden_size=8,
        num_key_value_heads=2,
        head_dim=4,
        gated_layers=[1, 3],
        num_cameras=NUM_CAMERAS,
        gate_init=0.5,
    )
    with torch.no_grad():
        bypass.camera_embedding[1] = 1.0
    patches = torch.randn(BATCH_SIZE, 64, 8)

    embs = [bypass.embed_patches(patches, camera_idx) for camera_idx in range(NUM_CAMERAS)]
    torch.testing.assert_close(embs[1] - embs[0], torch.ones_like(embs[0]))

    pad_masks = torch.ones(BATCH_SIZE, 32, dtype=torch.bool)
    spatial_kv = bypass(torch.cat(embs, dim=1), pad_masks)

    assert spatial_kv.keys.shape == (BATCH_SIZE, 32, 2, 4)
    assert spatial_kv.values.shape == (BATCH_SIZE, 32, 2, 4)
    assert spatial_kv.pad_masks is pad_masks
    # A fresh bypass contributes nothing: its value projection starts at zero
    assert torch.count_nonzero(spatial_kv.values) == 0
    assert torch.count_nonzero(spatial_kv.keys) > 0
    torch.testing.assert_close(bypass.gate(3), torch.tanh(torch.tensor(0.5)))
    with pytest.raises(KeyError):
        bypass.gate(2)


def test_spatial_bypass_rejects_zero_gate_init():
    with pytest.raises(ValueError, match="non-zero"):
        SpatialBypass(
            vision_hidden_size=4,
            num_key_value_heads=1,
            head_dim=4,
            gated_layers=[1],
            num_cameras=1,
            gate_init=0.0,
        )


def test_spatial_bypass_rejects_non_square_patch_grid():
    bypass = SpatialBypass(
        vision_hidden_size=4, num_key_value_heads=1, head_dim=4, gated_layers=[1], num_cameras=1
    )
    with pytest.raises(ValueError, match="square grid"):
        bypass.embed_patches(torch.randn(1, 10, 4), camera_idx=0)


def test_spatial_bypass_config_validation():
    with pytest.raises(ValueError, match="cross-attention"):
        SmolVLAConfig(spatial_bypass=True, attention_mode="self_attn")
    with pytest.raises(ValueError, match="spatial_pool_size"):
        SmolVLAConfig(spatial_bypass=True, spatial_pool_size=0)
    with pytest.raises(ValueError, match="spatial_gate_init"):
        SmolVLAConfig(spatial_bypass=True, spatial_gate_init=0.0)

    config = SmolVLAConfig(spatial_bypass=True, empty_cameras=1)
    config.input_features = {}
    with pytest.raises(ValueError, match="at least one image feature"):
        config.validate_features()

    config.input_features = {
        f"observation.images.cam_{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
        for i in range(NUM_CAMERAS)
    }
    config.validate_features()
    assert len(config.image_features) == NUM_CAMERAS + 1
    assert config.num_spatial_cameras == NUM_CAMERAS


def _make_policy(**config_kwargs):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig(
        max_action_dim=7,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        num_vlm_layers=4,
        num_steps=2,
        empty_cameras=1,
        **config_kwargs,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        **{
            f"observation.images.cam_{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
            for i in range(NUM_CAMERAS)
        },
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    # Instantiate policy on the GPU: initializing the VLM weights on the CPU is slow
    with torch.device("cuda"):
        return SmolVLAPolicy(config)


def _make_inputs(config: SmolVLAConfig):
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    size = config.resize_imgs_with_padding
    # Real cameras, then the empty camera as appended by `SmolVLAPolicy.prepare_images`
    images = [randn(BATCH_SIZE, 3, *size).clamp(-1, 1) for _ in range(NUM_CAMERAS)]
    images.append(-torch.ones_like(images[0]))
    img_masks = [torch.ones(BATCH_SIZE, dtype=torch.bool, device="cuda") for _ in range(NUM_CAMERAS)]
    img_masks.append(torch.zeros(BATCH_SIZE, dtype=torch.bool, device="cuda"))
    lang_tokens = torch.randint(0, 1000, (BATCH_SIZE, 8), device="cuda", generator=generator)
    lang_masks = torch.ones(BATCH_SIZE, 8, dtype=torch.bool, device="cuda")
    state = randn(BATCH_SIZE, config.max_state_dim)
    actions = randn(BATCH_SIZE, CHUNK_SIZE, config.max_action_dim)
    noise = randn(BATCH_SIZE, CHUNK_SIZE, config.max_action_dim)
    time = torch.full((BATCH_SIZE,), 0.5, device="cuda")
    return images, img_masks, lang_tokens, lang_masks, state, actions, noise, time


@pytest.fixture(scope="module")
def base_and_spatial_policies():
    # Fixtures are set up before the skip decorators of the tests run
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    set_seed(42)
    base = _make_policy()
    spatial = _make_policy(spatial_bypass=True)
    missing_keys, unexpected_keys = spatial.load_state_dict(base.state_dict(), strict=False)
    assert unexpected_keys == []
    assert missing_keys and all(".spatial_bypass." in key for key in missing_keys)
    assert spatial.model.vlm_with_expert.spatial_bypass.camera_embedding.shape[0] == NUM_CAMERAS
    return base.eval(), spatial.eval()


@skip_if_package_missing("transformers")
@require_cuda
def test_fresh_spatial_bypass_matches_base_policy(base_and_spatial_policies):
    base, spatial = base_and_spatial_policies
    images, img_masks, lang_tokens, lang_masks, state, actions, noise, time = _make_inputs(base.config)

    with torch.no_grad():
        torch.testing.assert_close(
            spatial.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time),
            base.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time),
        )
        torch.testing.assert_close(
            spatial.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise),
            base.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise),
        )


@skip_if_package_missing("transformers")
@require_cuda
def test_fresh_spatial_bypass_gets_value_gradients(base_and_spatial_policies):
    """The zero value projection must learn from the first step (zero gates would deadlock instead)."""
    _, spatial = base_and_spatial_policies
    bypass = spatial.model.vlm_with_expert.spatial_bypass
    assert bypass is not None
    images, img_masks, lang_tokens, lang_masks, state, actions, noise, time = _make_inputs(spatial.config)

    try:
        spatial.zero_grad()
        losses = spatial.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        losses.mean().backward()
        assert bypass.v_proj.weight.grad is not None and bypass.v_proj.weight.grad.abs().sum() > 0
    finally:
        spatial.zero_grad()


@skip_if_package_missing("transformers")
@require_cuda
def test_trained_spatial_bypass_changes_actions_and_trains(base_and_spatial_policies):
    base, spatial = base_and_spatial_policies
    bypass = spatial.model.vlm_with_expert.spatial_bypass
    assert bypass is not None
    images, img_masks, lang_tokens, lang_masks, state, actions, noise, time = _make_inputs(base.config)
    with torch.no_grad():
        expected = base.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

    try:
        with torch.no_grad():
            torch.nn.init.normal_(bypass.v_proj.weight, std=0.02)
            actions_with_bypass = spatial.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )
        assert not torch.allclose(actions_with_bypass, expected)

        spatial.zero_grad()
        losses = spatial.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        losses.mean().backward()
        for name, param in bypass.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0, name
    finally:
        spatial.zero_grad()
        with torch.no_grad():
            bypass.v_proj.weight.zero_()
