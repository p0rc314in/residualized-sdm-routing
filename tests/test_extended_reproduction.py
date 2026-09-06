from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from reproduction.recall_generate import (
    CONDITIONS, QUERIES, POINTER_KEYS, POINTER_HOPS, POINTER_QUERY_OFFSET,
    SPAN_MAP_OFFSET, SPAN_QUERY_OFFSET, OVERWRITE_MAP_OFFSET, OVERWRITE_QUERY_OFFSET,
    MAXIMUM_SPAN, SPAN_VALUES, generate_batch, prepare_dataset,
)
from reproduction.recall_data import AdaptiveRecallData
from reproduction.recall_model import AdaptiveRecallEmbedding
from reproduction.babylm.metrics import terminal_metrics
from reproduction.babylm.spec import PUBLIC_ARMS, SPEC, CHECKPOINT_INPUT_SHA256
from reproduction.babylm.data import BabyLMData
from reproduction.spec import ARM_ORDER
from scripts.check_recall_reproduction import validate_result
from scripts.check_babylm_reproduction import validate_training
from scripts.run_reproductions import arm_commands, environment
from scripts.verify_vendor import validate_babylm_vendor

ROOT = Path(__file__).resolve().parents[1]


class ExtendedReproductionTests(unittest.TestCase):
    def test_recall_labels_match_independent_symbolic_solver(self):
        generator = np.random.default_rng(102337)
        for condition in CONDITIONS:
            inputs, labels = generate_batch(generator, 3, condition)
            for tokens, expected in zip(inputs.tolist(), labels.tolist()):
                memory, queries = tokens[:-QUERIES], tokens[-QUERIES:]
                if condition["family"] == "pointer_chase":
                    table = dict(divmod(token, POINTER_KEYS) for token in memory)
                    actual = []
                    for query in queries:
                        hop_index, value = divmod(query - POINTER_QUERY_OFFSET, POINTER_KEYS)
                        for _ in range(POINTER_HOPS[hop_index]):
                            value = table[value]
                        actual.append(value)
                else:
                    overwrite = condition["family"] == "overwrite_recall"
                    mapping_offset = OVERWRITE_MAP_OFFSET if overwrite else SPAN_MAP_OFFSET
                    query_offset = OVERWRITE_QUERY_OFFSET if overwrite else SPAN_QUERY_OFFSET
                    table = {}
                    for token in memory:
                        key_slot, value = divmod(token - mapping_offset, SPAN_VALUES)
                        table[key_slot] = value
                    actual = [table[token - query_offset] for token in queries]
                self.assertEqual(actual, expected, condition["id"])

    def test_ragged_recall_loader_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = prepare_dataset(root / "data", steps=3, batch_size=2, eval_examples=4,
                                   stream_seed=102337, progress_every=0)
            data = AdaptiveRecallData(path)
            for index in range(3):
                inputs, labels, condition = data.train_batch(index)
                self.assertEqual(inputs.shape, (2, condition["sequence_length"]))
                self.assertEqual(labels.shape, (2, 16))
            for condition in CONDITIONS:
                inputs, labels = data.evaluation("test", condition["id"])
                self.assertEqual(inputs.shape, (4, condition["sequence_length"]))
                self.assertEqual(labels.shape, (4, 16))
            token_path = path.parent / data.manifest["records"]["train_tokens"]["path"]
            with token_path.open("r+b") as handle:
                handle.write(b"bad!")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                AdaptiveRecallData(path)

    def test_semantic_embedding_accepts_every_condition(self):
        import torch
        embedding = AdaptiveRecallEmbedding(128)
        for condition in CONDITIONS:
            tokens, _ = generate_batch(np.random.default_rng(4), 1, condition)
            values = embedding(torch.from_numpy(tokens.astype(np.int64)))
            self.assertEqual(tuple(values.shape), (1, condition["sequence_length"], 128))
            self.assertTrue(torch.isfinite(values).all())

    def test_five_recall_and_three_babylm_commands_keep_access_explicit(self):
        for benchmark, arms, width in (("adaptive_recall", ARM_ORDER, "8"), ("babylm", PUBLIC_ARMS, "32")):
            for arm in arms:
                command, = arm_commands(benchmark, "smoke", arm, Path("inputs"), Path("smoke"), smoke_steps=2)
                self.assertEqual(command[command.index("--reads") + 1], width)
                self.assertEqual(command[command.index("--writes") + 1], width)
                self.assertEqual(command[command.index("--smoke-steps") + 1], "2")
                command, = arm_commands(benchmark, "train", arm, Path("inputs"), Path("results"))
                self.assertNotIn("--smoke-steps", command)

    def test_babylm_runtime_is_isolated_and_byte_pinned(self):
        validate_babylm_vendor(ROOT / "third_party/babylm_runtime")
        self.assertNotIn(str(ROOT / "third_party/runtime"), environment("babylm")["PYTHONPATH"].split(":"))
        self.assertNotIn(str(ROOT / "third_party/babylm_runtime"), environment("adaptive_recall")["PYTHONPATH"].split(":"))

    def test_smoke_cannot_satisfy_result_verifiers(self):
        with self.assertRaisesRegex(ValueError, "incomplete or changed"):
            validate_result({"status": "smoke_passed"}, "dense_attention")
        with self.assertRaisesRegex(ValueError, "not complete"):
            validate_training({"status": "smoke_passed"}, {}, [], "a16_dense_attention")

    def test_babylm_manifest_is_mandatory_before_loading_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text('{"format":"changed"}')
            with self.assertRaisesRegex(ValueError, "recorded experiment"):
                BabyLMData(path, root)

    def test_six_tasks_and_primary_metric_macro(self):
        tasks = {name: {"accuracy": .5} for name in
                 ("blimp", "blimp_supplement", "comps", "entity_tracking", "ewok_fast")}
        tasks.update({"global_piqa_parallel": {"accuracy": .2}, "global_piqa_nonparallel": {"accuracy": .6}})
        fine = {name: {"selection_metric": "f1" if name in ("mrpc", "qqp") else "accuracy",
                       "best_validation": {"accuracy": .3, "f1": .9}}
                for name in ("boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc")}
        result = terminal_metrics({"zero_shot": {"tasks": tasks}, "finetune": fine}, {"ewok": {"accuracy": .8}})
        self.assertEqual(len(result["zero_shot"]), 6)
        self.assertAlmostEqual(result["zero_shot_macro"], (4 * .5 + .8 + .4) / 6)
        self.assertAlmostEqual(result["finetune_macro_primary_metric"], (5 * .3 + 2 * .9) / 7)
        self.assertNotIn("ewok_fast", result["zero_shot"])

    def test_babylm_training_schedule_and_checkpoint_identity(self):
        metadata = json.loads((ROOT / "data/babylm-input-manifest.json").read_text())
        training = metadata["corpus"]["training"]
        self.assertEqual(training["optimizer_steps"], 102852)
        self.assertEqual(training["target_token_presentations"], 1685114880)
        self.assertEqual((SPEC.reads, SPEC.writes), (32, 32))
        self.assertEqual(len(training["checkpoint_exposures"]), 28)
        self.assertEqual(training["checkpoint_exposures"][-1]["optimizer_step"], 102852)
        self.assertEqual(len(CHECKPOINT_INPUT_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
