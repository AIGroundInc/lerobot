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

"""Test SmolVLA-Control (CNN control branch FiLM-conditioning the SmolVLA action expert)."""

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla_control.configuration_smolvla_control import SmolVLAControlConfig
from lerobot.policies.smolvla_control.control_branch import ControlBranch
from lerobot.policies.smolvla_control.convert_smolvla_checkpoint import convert_smolvla_checkpoint
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, skip_if_package_missing

BATCH_SIZE = 2
CHUNK_SIZE = 10
NUM_CAMERAS = 2
# Random backbone with GroupNorm: no weight download and no batch statistics in the tests
TEST_CONTROL_KWARGS = {
    "control_pretrained_backbone_weights": None,
    "control_use_group_norm": True,
    "control_image_size": (64, 64),
    "control_num_keypoints": 4,
}


def _input_features(num_cameras: int = NUM_CAMERAS) -> dict[str, PolicyFeature]:
    return {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        **{
            f"observation.images.cam_{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
            for i in range(num_cameras)
        },
    }


def test_registration():
    config = SmolVLAControlConfig()
    assert config.type == "smolvla_control"
    assert isinstance(config, SmolVLAConfig)
    assert get_policy_class("smolvla_control").__name__ == "SmolVLAControlPolicy"


def test_config_validation():
    with pytest.raises(ValueError, match="ResNet"):
        SmolVLAControlConfig(control_vision_backbone="vit_b_16")
    with pytest.raises(ValueError, match="control_use_group_norm"):
        SmolVLAControlConfig(control_use_group_norm=True)
    with pytest.raises(ValueError, match="control_image_size"):
        SmolVLAControlConfig(control_image_size=(0, 64))
    with pytest.raises(ValueError, match="control_num_keypoints"):
        SmolVLAControlConfig(control_num_keypoints=0)
    with pytest.raises(ValueError, match="control_num_cameras"):
        SmolVLAControlConfig(control_num_cameras=0)

    config = SmolVLAControlConfig(empty_cameras=1)
    config.input_features = {}
    with pytest.raises(ValueError, match="at least one image feature"):
        config.validate_features()

    config.input_features = _input_features()
    config.validate_features()
    assert len(config.image_features) == NUM_CAMERAS + 1
    assert config.num_control_cameras == NUM_CAMERAS

    # e.g. smolvla_base declares 3 cameras, LIBERO only provides 2
    config = SmolVLAControlConfig(control_num_cameras=2)
    config.input_features = _input_features(num_cameras=3)
    config.validate_features()
    assert config.num_control_cameras == 2
    config = SmolVLAControlConfig(control_num_cameras=4)
    config.input_features = _input_features(num_cameras=3)
    with pytest.raises(ValueError, match="exceeds"):
        config.validate_features()


def test_convert_smolvla_checkpoint(tmp_path):
    source = tmp_path / "smolvla"
    smolvla_config = SmolVLAConfig(chunk_size=CHUNK_SIZE, n_action_steps=CHUNK_SIZE)
    smolvla_config.input_features = _input_features()
    source.mkdir()
    smolvla_config._save_pretrained(source)
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "policy_preprocessor.json").write_text("{}")
    (source / "train_config.json").write_text("{}")

    destination = convert_smolvla_checkpoint(str(source), tmp_path / "control")

    assert sorted(path.name for path in destination.iterdir()) == [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
    ]
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    config = PreTrainedConfig.from_pretrained(destination)
    assert isinstance(config, SmolVLAControlConfig)
    assert config.chunk_size == CHUNK_SIZE
    assert config.control_vision_backbone == SmolVLAControlConfig.control_vision_backbone

    with pytest.raises(FileExistsError):
        convert_smolvla_checkpoint(str(source), destination)
    with pytest.raises(ValueError, match="'smolvla' checkpoint"):
        convert_smolvla_checkpoint(str(destination), tmp_path / "again")


def _make_branch(share_encoder: bool, hidden_size: int = 16, num_layers: int = 3) -> ControlBranch:
    config = SmolVLAControlConfig(empty_cameras=1, control_share_encoder=share_encoder, **TEST_CONTROL_KWARGS)
    config.input_features = _input_features()
    config.validate_features()
    return ControlBranch(config, hidden_size=hidden_size, num_layers=num_layers)


def _branch_inputs(size: int = 96) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    images = [torch.rand(BATCH_SIZE, 3, size, size) * 2 - 1 for _ in range(NUM_CAMERAS)]
    images.append(-torch.ones_like(images[0]))  # empty camera, ignored
    img_masks = [torch.ones(BATCH_SIZE, dtype=torch.bool) for _ in range(NUM_CAMERAS)]
    img_masks.append(torch.zeros(BATCH_SIZE, dtype=torch.bool))
    return images, img_masks


@pytest.mark.parametrize("share_encoder", [True, False])
def test_control_branch_shapes_and_zero_init(share_encoder):
    set_seed(0)
    branch = _make_branch(share_encoder)
    assert len(branch.encoders) == (1 if share_encoder else NUM_CAMERAS)
    images, img_masks = _branch_inputs()

    keypoints = branch.encode(images, img_masks)
    film = branch(images, img_masks)

    assert keypoints.shape == (BATCH_SIZE, NUM_CAMERAS * 2 * 4)
    assert film.shape == (BATCH_SIZE, 3, 4, 16)
    # A fresh branch does not modulate the expert
    assert torch.count_nonzero(film) == 0


def test_control_branch_zeroes_missing_cameras():
    branch = _make_branch(share_encoder=True)
    images, img_masks = _branch_inputs()
    img_masks[1] = torch.tensor([True, False])

    keypoints = branch.encode(images, img_masks).view(BATCH_SIZE, NUM_CAMERAS, -1)

    assert torch.count_nonzero(keypoints[1, 1]) == 0
    assert torch.count_nonzero(keypoints[0, 1]) > 0
    assert torch.count_nonzero(keypoints[:, 0]) > 0


def test_control_branch_learns_from_the_first_step():
    """The zero FiLM projection gets gradients right away (no zero-gate deadlock), then the rest follows."""
    branch = _make_branch(share_encoder=True)
    images, img_masks = _branch_inputs()

    branch(images, img_masks).sum().backward()
    assert branch.film_proj.weight.grad is not None and branch.film_proj.weight.grad.abs().sum() > 0

    branch.zero_grad()
    with torch.no_grad():
        torch.nn.init.normal_(branch.film_proj.weight, std=0.02)
    branch(images, img_masks).sum().backward()
    for name, param in branch.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name


def _make_policy(policy_type: str):
    config_kwargs = {
        "max_action_dim": 7,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": CHUNK_SIZE,
        "num_vlm_layers": 4,
        "num_steps": 2,
        "empty_cameras": 1,
    }
    if policy_type == "smolvla_control":
        config: SmolVLAConfig = SmolVLAControlConfig(**config_kwargs, **TEST_CONTROL_KWARGS)
    else:
        config = SmolVLAConfig(**config_kwargs)
    config.input_features = _input_features()
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    # Instantiate policy on the GPU: initializing the VLM weights on the CPU is slow
    with torch.device("cuda"):
        # `.to` also moves the CPU buffers SpatialSoftmax builds from numpy, as `make_policy` does
        return get_policy_class(policy_type)(config).to("cuda")


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
def base_and_control_policies():
    # Fixtures are set up before the skip decorators of the tests run
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    set_seed(42)
    base = _make_policy("smolvla")
    control = _make_policy("smolvla_control")
    # Loading a SmolVLA state dict leaves only the control branch missing
    missing_keys, unexpected_keys = control.load_state_dict(base.state_dict(), strict=False)
    assert unexpected_keys == []
    assert missing_keys and all(key.startswith("model.control_branch.") for key in missing_keys)
    return base.eval(), control.eval()


@skip_if_package_missing("transformers")
@require_cuda
def test_fresh_control_branch_matches_base_policy(base_and_control_policies):
    base, control = base_and_control_policies
    images, img_masks, lang_tokens, lang_masks, state, actions, noise, time = _make_inputs(base.config)

    with torch.no_grad():
        torch.testing.assert_close(
            control.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time),
            base.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time),
        )
        torch.testing.assert_close(
            control.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise),
            base.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise),
        )
    assert control.model._film is None


@skip_if_package_missing("transformers")
@require_cuda
def test_fresh_control_branch_gets_film_gradients(base_and_control_policies):
    _, control = base_and_control_policies
    branch = control.model.control_branch
    images, img_masks, lang_tokens, lang_masks, state, actions, noise, time = _make_inputs(control.config)

    try:
        control.zero_grad()
        losses = control.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        losses.mean().backward()
        assert branch.film_proj.weight.grad is not None and branch.film_proj.weight.grad.abs().sum() > 0
    finally:
        control.zero_grad()


@skip_if_package_missing("transformers")
@require_cuda
def test_trained_control_branch_changes_actions_and_trains(base_and_control_policies):
    base, control = base_and_control_policies
    branch = control.model.control_branch
    images, img_masks, lang_tokens, lang_masks, state, actions, noise, time = _make_inputs(base.config)
    with torch.no_grad():
        expected = base.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

    try:
        with torch.no_grad():
            torch.nn.init.normal_(branch.film_proj.weight, std=0.02)
            actions_with_film = control.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )
        assert not torch.allclose(actions_with_film, expected)

        control.zero_grad()
        losses = control.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        losses.mean().backward()
        for name, param in branch.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0, name
    finally:
        control.zero_grad()
        with torch.no_grad():
            branch.film_proj.weight.zero_()


@skip_if_package_missing("transformers")
@require_cuda
def test_default_peft_targets_train_the_control_branch(base_and_control_policies):
    _, control = base_and_control_policies
    assert control._get_default_peft_targets()["modules_to_save"] == ["control_branch"]


def test_control_branch_rejects_missing_cameras_with_a_hint():
    config = SmolVLAControlConfig(**TEST_CONTROL_KWARGS)
    config.input_features = _input_features(num_cameras=3)
    config.validate_features()
    branch = ControlBranch(config, hidden_size=16, num_layers=3)
    images, img_masks = _branch_inputs()  # 2 cameras (+ an empty one), as `prepare_images` drops camera 3

    with pytest.raises(ValueError, match="control_num_cameras"):
        branch(images[:NUM_CAMERAS], img_masks[:NUM_CAMERAS])
