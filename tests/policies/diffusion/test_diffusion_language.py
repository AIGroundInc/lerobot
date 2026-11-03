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

"""Test language conditioning of Diffusion Policy (task instruction encoded by a frozen CLIP text model)."""

import pytest
import torch

pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.random_utils import set_seed  # noqa: E402

BATCH_SIZE = 2
N_OBS_STEPS = 2
HORIZON = 16
SEQ_LEN = 8
IMAGE_KEY = "observation.images.cam"


class _TinyCLIPTextModel:
    """Stands in for `CLIPTextModel` so the tests don't download CLIP."""

    @staticmethod
    def from_pretrained(_name):
        config = transformers.CLIPTextConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=SEQ_LEN,
        )
        model = transformers.CLIPTextModel(config)
        # Like transformers' checkpoint loading, make q/k/v views of one fused tensor
        attention = model.text_model.encoder.layers[0].self_attn
        projections = [attention.q_proj, attention.k_proj, attention.v_proj]
        fused = torch.cat([proj.weight.data for proj in projections])
        for proj, view in zip(projections, fused.chunk(3), strict=True):
            proj.weight.data = view
        return model


@pytest.fixture
def policy(monkeypatch):
    from lerobot.policies.multi_task_dit import modeling_multi_task_dit

    monkeypatch.setattr(modeling_multi_task_dit, "CLIPTextModel", _TinyCLIPTextModel)
    set_seed(0)
    config = DiffusionConfig(
        use_language=True,
        text_embed_dim=8,
        n_obs_steps=N_OBS_STEPS,
        horizon=HORIZON,
        n_action_steps=8,
        down_dims=(16, 32),
        pretrained_backbone_weights=None,
        use_group_norm=True,
        noise_scheduler_type="DDIM",
        num_inference_steps=2,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    return DiffusionPolicy(config)


def _language(token_offset: int = 0) -> dict[str, torch.Tensor]:
    tokens = (torch.arange(SEQ_LEN) + token_offset).repeat(BATCH_SIZE, 1)
    return {
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones_like(tokens, dtype=torch.bool),
    }


def _observation(n_obs_steps: int | None = None) -> dict[str, torch.Tensor]:
    steps = () if n_obs_steps is None else (n_obs_steps,)
    return {
        OBS_STATE: torch.randn(BATCH_SIZE, *steps, 4),
        IMAGE_KEY: torch.rand(BATCH_SIZE, *steps, 3, 32, 32),
    }


def test_text_features_extend_the_global_conditioning(policy):
    text_encoder = policy.diffusion.text_encoder
    assert text_encoder is not None
    assert all(not p.requires_grad for p in text_encoder.text_encoder.parameters())

    batch = {
        **_observation(N_OBS_STEPS),
        **_language(),
        ACTION: torch.randn(BATCH_SIZE, HORIZON, 2),
        "action_is_pad": torch.zeros(BATCH_SIZE, HORIZON, dtype=torch.bool),
    }
    loss, _ = policy.forward(batch)
    loss.backward()

    assert text_encoder.projection.weight.grad is not None
    assert text_encoder.projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in text_encoder.text_encoder.parameters())


def test_actions_depend_on_the_instruction(policy):
    policy.eval()
    observation = {**_observation(N_OBS_STEPS)}
    noise = torch.randn(BATCH_SIZE, HORIZON, 2)

    actions_a = policy.predict_action_chunk({**observation, **_language(0)}, noise=noise)
    actions_b = policy.predict_action_chunk({**observation, **_language(50)}, noise=noise)

    assert actions_a.shape == (BATCH_SIZE, 8, 2)
    assert not torch.allclose(actions_a, actions_b)


def test_policy_round_trips_through_safetensors(policy, tmp_path):
    policy.save_pretrained(tmp_path)
    reloaded = DiffusionPolicy.from_pretrained(tmp_path)

    for (name, param), reloaded_param in zip(
        policy.state_dict().items(), reloaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(param, reloaded_param.to(param.device), msg=name)


def test_select_action_keeps_the_instruction_with_queued_observations(policy):
    policy.eval()
    policy.reset()

    # Once observations are queued, `predict_action_chunk` rebuilds the batch from the queues; the
    # instruction is not queued and must still reach the model.
    action = policy.select_action({**_observation(), **_language()})

    assert action.shape == (BATCH_SIZE, 2)
