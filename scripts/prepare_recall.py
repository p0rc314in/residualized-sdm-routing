#!/usr/bin/env python3
"""Generate the pinned Recall stream twice and reject any byte mismatch."""
from pathlib import Path
import argparse
import shutil
import tempfile

import sys
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party/runtime"))

from reproduction.recall_generate import prepare_dataset
from reproduction.recall_data import AdaptiveRecallData
from reproduction.recall_model import validate_codec
from reproduction.recall_spec import MANIFEST_SHA256, STEPS, STREAM_SEED, BATCH_SIZE, EVAL_EXAMPLES
from reproduction.wikitext103_coverage import sha256_file


def check(path: Path) -> AdaptiveRecallData:
    if sha256_file(path) != MANIFEST_SHA256:
        raise ValueError("Recall manifest differs from the recorded experiment")
    data = AdaptiveRecallData(path)
    validate_codec(data.manifest)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.check or (output / "manifest.json").is_file():
        check(output / "manifest.json")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".recall-build-") as temporary:
            builds = [Path(temporary) / str(index) for index in range(2)]
            for build in builds:
                manifest = prepare_dataset(build, steps=STEPS, batch_size=BATCH_SIZE,
                                           eval_examples=EVAL_EXAMPLES, stream_seed=STREAM_SEED)
                check(manifest)
            # The pinned manifest contains the hash of every generated record.
            shutil.move(str(builds[0]), output)
    print(f"Recall input verified: {MANIFEST_SHA256}")


if __name__ == "__main__":
    main()
