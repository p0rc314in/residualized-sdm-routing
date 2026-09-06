"""Checksum-verified readers for the two immutable M5 quality streams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from reproduction.wikitext103_coverage import sha256_file


class ManifestArrays:
    _DTYPES = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.arrays: dict[str, np.memmap] = {}
        self.hashes: dict[str, str] = {}

    def load_record(self, name: str, row: dict[str, Any]) -> np.memmap:
        path = (self.root / str(row["path"])).resolve()
        if self.root not in path.parents:
            raise ValueError("record escapes input root")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"size mismatch for {path}")
        digest = sha256_file(path)
        if digest != str(row["sha256"]):
            raise ValueError(
                f"SHA-256 mismatch for {path}: {digest} != {row['sha256']}"
            )
        dtype_name = str(row["dtype"])
        if dtype_name not in self._DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r}")
        self.hashes[name] = digest
        value = np.memmap(
            path,
            mode="r",
            dtype=self._DTYPES[dtype_name],
            shape=tuple(int(item) for item in row["shape"]),
        )
        self.arrays[name] = value
        return value


class AdaptiveRecallData(ManifestArrays):
    """Immutable ragged exact-recall stream used by the accepted SDM studies."""

    FORMAT = "elastic_sdm_adaptive_recall_suite_v1"

    def __init__(self, manifest_path: Path) -> None:
        super().__init__(manifest_path)
        if self.manifest.get("format") != self.FORMAT:
            raise ValueError("unexpected adaptive-recall stream format")
        self.conditions = tuple(self.manifest["conditions"])
        self.condition_by_id = {str(row["id"]): row for row in self.conditions}
        if len(self.condition_by_id) != len(self.conditions):
            raise ValueError("condition identifiers are not unique")
        for name, row in self.manifest["records"].items():
            self.load_record(name, row)
        self.train_tokens = self.arrays["train_tokens"]
        self.train_labels = self.arrays["train_labels"]
        self.train_condition_ids = self.arrays["train_condition_ids"]
        self.train_offsets = self.arrays["train_token_offsets"]

    def condition(self, identifier: str | int) -> dict[str, Any]:
        if isinstance(identifier, str):
            if identifier not in self.condition_by_id:
                raise ValueError(f"unknown condition: {identifier}")
            return self.condition_by_id[identifier]
        index = int(identifier)
        if index < 0 or index >= len(self.conditions):
            raise ValueError(f"condition index out of range: {index}")
        return self.conditions[index]

    def train_batch(self, step: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        condition = self.condition(int(self.train_condition_ids[step]))
        length = int(condition["sequence_length"])
        start = int(self.train_offsets[step])
        stop = int(self.train_offsets[step + 1])
        tokens = np.asarray(self.train_tokens[start:stop]).reshape(
            int(self.manifest["batch_size"]), length
        )
        return tokens, np.asarray(self.train_labels[step]), condition

    def evaluation(
        self, split: str, identifier: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        if split not in ("validation", "test"):
            raise ValueError(f"unsupported split: {split}")
        condition = self.condition(identifier)
        index = int(condition["index"])
        offsets = self.arrays[f"{split}_token_offsets"]
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        tokens = np.asarray(self.arrays[f"{split}_tokens"][start:stop]).reshape(
            int(self.manifest["eval_examples"]),
            int(condition["sequence_length"]),
        )
        return tokens, np.asarray(self.arrays[f"{split}_labels"][index])


__all__ = ["AdaptiveRecallData"]
