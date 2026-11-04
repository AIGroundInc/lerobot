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

"""Turn a SmolVLA checkpoint into a SmolVLA-Control checkpoint to fine-tune with `--policy.path`.

`--policy.path` takes the policy type from the checkpoint's `config.json`, so a SmolVLA checkpoint always builds
a SmolVLA policy. This copies the checkpoint (weights and processors) and only switches its type. Loading it
builds the control branch from the config defaults (override them with `--policy.control_*`): the SmolVLA
weights load as usual and the branch, missing from the checkpoint, starts from its initialization, whose zero
FiLM projection reproduces the SmolVLA policy.

    uv run python -m lerobot.policies.smolvla_control.convert_smolvla_checkpoint \
        lerobot/smolvla_libero outputs/smolvla_control_libero_init
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import CONFIG_NAME

# Files describing the source run rather than the converted model
SKIPPED_FILES = {CONFIG_NAME, "README.md", "train_config.json", ".gitattributes"}


def convert_smolvla_checkpoint(source: str, destination: Path, revision: str | None = None) -> Path:
    """Copy the SmolVLA checkpoint `source` (local directory or Hub repo id) to `destination` as SmolVLA-Control."""
    source_dir = Path(source)
    if not source_dir.is_dir():
        source_dir = Path(snapshot_download(source, revision=revision))
    config = json.loads((source_dir / CONFIG_NAME).read_text())
    if config.get("type") != "smolvla":
        raise ValueError(f"Expected a 'smolvla' checkpoint, got type {config.get('type')!r} in {source}.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} already exists and is not empty.")

    destination.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file() and path.name not in SKIPPED_FILES:
            # Resolve the symlinks of the Hub cache
            shutil.copyfile(path.resolve(), destination / path.name)
    config["type"] = "smolvla_control"
    (destination / CONFIG_NAME).write_text(json.dumps(config, indent=4) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="SmolVLA checkpoint: local directory or Hub repo id")
    parser.add_argument(
        "destination", type=Path, help="New (or empty) directory for the converted checkpoint"
    )
    parser.add_argument("--revision", default=None, help="Hub revision of the source checkpoint")
    args = parser.parse_args()
    destination = convert_smolvla_checkpoint(args.source, args.destination, revision=args.revision)
    print(f"Wrote the SmolVLA-Control checkpoint to {destination}")


if __name__ == "__main__":
    main()
