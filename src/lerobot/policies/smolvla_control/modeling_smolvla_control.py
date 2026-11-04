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

"""SmolVLA-Control: SmolVLA whose action expert is FiLM-conditioned on a trainable CNN control branch.

```
              VLM K/V (semantic, unchanged)
                         │
noisy actions ─► RMSNorm ─► FiLM ─► attention ─► + ─► RMSNorm ─► FiLM ─► MLP ─► + ─► ... (every expert layer)
                            ▲                                     ▲
cameras ─► ResNet ─► SpatialSoftmax ─► MLP ─► per-layer (scale, shift)
```

The FiLM is applied through forward hooks on the expert's `input_layernorm` and `post_attention_layernorm`,
so the SmolVLA attention code and the checkpoint keys stay untouched. The FiLM parameters are computed once
per observation and reused by every denoising step.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, Unpack

from torch import Tensor, nn

from ..pretrained import RTCActionSelectKwargs
from ..rtc.modeling_rtc import RTCProcessor
from ..smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from .configuration_smolvla_control import SmolVLAControlConfig
from .control_branch import ControlBranch

ATTENTION_SLOT = 0
MLP_SLOT = 1


class ControlVLAFlowMatching(VLAFlowMatching):
    def __init__(self, config: SmolVLAControlConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__(config, rtc_processor=rtc_processor)
        expert_layers = self.vlm_with_expert.lm_expert.layers
        self.control_branch = ControlBranch(
            config, hidden_size=self.vlm_with_expert.expert_hidden_size, num_layers=len(expert_layers)
        )
        # [B, num_layers, 4, hidden] while a forward pass or a sampling runs, None otherwise
        self._film: Tensor | None = None
        for layer_idx, layer in enumerate(expert_layers):
            layer.input_layernorm.register_forward_hook(
                partial(self._modulate, layer_idx=layer_idx, slot=ATTENTION_SLOT)
            )
            layer.post_attention_layernorm.register_forward_hook(
                partial(self._modulate, layer_idx=layer_idx, slot=MLP_SLOT)
            )

    def _modulate(
        self, module: nn.Module, args: tuple[Any, ...], output: Tensor, *, layer_idx: int, slot: int
    ) -> Tensor | None:
        """Forward hook on an expert norm: h -> (1 + scale) * h + shift."""
        if self._film is None:
            return None
        scale = self._film[:, layer_idx, 2 * slot, None, :].to(output.dtype)
        shift = self._film[:, layer_idx, 2 * slot + 1, None, :].to(output.dtype)
        return output * (1.0 + scale) + shift

    @contextmanager
    def _conditioned_on(self, images: list[Tensor], img_masks: list[Tensor]) -> Iterator[None]:
        self._film = self.control_branch(images, img_masks)
        try:
            yield
        finally:
            self._film = None

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        with self._conditioned_on(images, img_masks):
            return super().forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        with self._conditioned_on(images, img_masks):
            return super().sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise, **kwargs)


class SmolVLAControlPolicy(SmolVLAPolicy):
    """SmolVLA with a DP-style control vision branch FiLM-conditioning the action expert."""

    config_class = SmolVLAControlConfig
    name = "smolvla_control"

    def _build_model(self) -> ControlVLAFlowMatching:
        if not isinstance(self.config, SmolVLAControlConfig):
            raise TypeError(f"Expected a SmolVLAControlConfig, got {type(self.config).__name__}.")
        return ControlVLAFlowMatching(self.config, rtc_processor=self.rtc_processor)

    def _get_default_peft_targets(self) -> dict[str, Any]:
        targets = super()._get_default_peft_targets()
        # The branch is new (its FiLM projection starts at zero), so train it fully
        targets["modules_to_save"] = [*targets["modules_to_save"], "control_branch"]
        return targets
