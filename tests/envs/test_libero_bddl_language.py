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

import pytest

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

pytest.importorskip("libero.libero", reason="LIBERO is required (install lerobot[libero])")

from lerobot.envs.libero import read_bddl_language  # noqa: E402


def test_read_bddl_language_normalizes_whitespace(tmp_path):
    bddl_file = tmp_path / "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
    bddl_file.write_text(
        "(define (problem LIBERO_Tabletop_Manipulation)\n"
        "  (:domain robosuite)\n"
        "  (:language grab alphabet soup\n    and put it into basket)\n"
        "  (:regions)\n)"
    )

    assert read_bddl_language(bddl_file) == "grab alphabet soup and put it into basket"


def test_read_bddl_language_requires_language_field(tmp_path):
    bddl_file = tmp_path / "task.bddl"
    bddl_file.write_text("(define (problem LIBERO_Tabletop_Manipulation)\n  (:domain robosuite)\n)")

    with pytest.raises(ValueError, match="language"):
        read_bddl_language(bddl_file)


def test_language_from_bddl_is_forwarded_to_the_env():
    assert LiberoEnvConfig().gym_kwargs["language_from_bddl"] is False
    assert LiberoEnvConfig(language_from_bddl=True).gym_kwargs["language_from_bddl"] is True
