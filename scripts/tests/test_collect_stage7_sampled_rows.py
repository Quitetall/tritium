import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "collect-stage7-sampled-rows.py"


def load_module():
    spec = importlib.util.spec_from_file_location("collect_stage7_sampled_rows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixtureSource:
    def __init__(self, module, root: Path, *, duplicate_only: bool = False):
        self.module = module
        self.root = root
        self.paths = {}
        self.shards = {}
        self.materialized = []
        self.preflighted = []
        for contract in module.DATASETS:
            suffix = ".json.gz" if contract.repo_id == "allenai/c4" else ".parquet"
            path = root / f"{contract.repo_id.replace('/', '--')}{suffix}"
            values = [
                f"{contract.repo_id} immutable source row {index} with unique payload"
                for index in range(48)
            ]
            if duplicate_only:
                values = [values[0]] * len(values)
            if suffix == ".json.gz":
                with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
                    for value in values:
                        stream.write(json.dumps({contract.text_field: value}) + "\n")
            else:
                pq.write_table(pa.table({contract.text_field: values}), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            shard = module.SourceShard(
                path=f"{contract.data_dir or 'data'}/train-00000{suffix}",
                bytes=path.stat().st_size,
                sha256=digest,
                hub_oid=f"sha256:{digest}",
            )
            self.paths[(contract.repo_id, shard.path)] = path
            self.shards[contract.repo_id] = (shard,)

    def list_shards(self, contract):
        return self.shards[contract.repo_id]

    def materialize(self, contract, shard):
        self.materialized.append(contract.repo_id)
        return self.paths[(contract.repo_id, shard.path)]

    def preflight(self, contract, shard):
        self.preflighted.append(contract.repo_id)


class DeniedFinalSource(FixtureSource):
    def preflight(self, contract, shard):
        super().preflight(contract, shard)
        if contract.repo_id == "bigcode/starcoderdata":
            raise PermissionError("gated source access denied")


class RacingOutputSource(FixtureSource):
    def __init__(self, module, root: Path, output: Path):
        super().__init__(module, root)
        self.output = output

    def materialize(self, contract, shard):
        if not self.output.exists():
            self.output.mkdir()
        return super().materialize(contract, shard)


class CollectStage7SampledRowsTests(unittest.TestCase):
    def test_collects_disjoint_content_bound_rows_and_refuses_overwrite(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = FixtureSource(module, root)
            output = root / "sampled"
            module.collect_sampled_rows(
                output,
                source,
                minimum_utf8_bytes={contract.repo_id: 96 for contract in module.DATASETS},
                minimum_rows={contract.repo_id: 2 for contract in module.DATASETS},
            )

            manifest = json.loads((output / "sampled-rows.json").read_bytes())
            acquisition = json.loads((output / "acquisition-receipt.json").read_bytes())
            self.assertEqual(manifest["schema"], "tritium.stage7-sampled-rows.v1")
            self.assertEqual(acquisition["schema"], "tritium.stage7-row-acquisition.v1")
            self.assertTrue(acquisition["receipt_id"].startswith("sha256:"))
            unsigned = {key: value for key, value in acquisition.items() if key != "receipt_id"}
            self.assertEqual(
                acquisition["receipt_id"],
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        unsigned,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            )
            self.assertEqual(
                acquisition["sampled_rows"]["sha256"],
                hashlib.sha256((output / "sampled-rows.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                acquisition["collection_policy"]["order"],
                "seeded-partition-ranking-over-lexicographic-shard-row-v1",
            )
            self.assertEqual(
                acquisition["collection_policy"]["minimum_utf8_bytes"],
                {contract.repo_id: 96 for contract in module.DATASETS},
            )
            self.assertEqual(
                acquisition["collection_policy"]["partition_order"],
                ["calibration", "refinement", "validation", "evaluation"],
            )
            self.assertEqual(
                acquisition["collection_policy"]["minimum_rows"],
                {contract.repo_id: 2 for contract in module.DATASETS},
            )
            self.assertEqual(
                list(manifest["partitions"]),
                ["calibration", "refinement", "validation", "evaluation"],
            )
            self.assertEqual(
                {
                    partition: [
                        row["row_index"]
                        for row in acquisition["partitions"][partition]["datasets"][0][
                            "rows"
                        ]
                    ]
                    for partition in module.PARTITIONS
                },
                {
                    "calibration": [1, 3],
                    "refinement": [0, 6],
                    "validation": [2, 5],
                    "evaluation": [4, 7],
                },
            )

            locators = set()
            content = set()
            for partition in manifest["partitions"].values():
                self.assertEqual(len(partition["datasets"]), 3)
                for dataset in partition["datasets"]:
                    rows_path = output / dataset["rows"]["path"]
                    self.assertEqual(
                        hashlib.sha256(rows_path.read_bytes()).hexdigest(),
                        dataset["rows"]["sha256"],
                    )
                    for line in rows_path.read_text().splitlines():
                        row = json.loads(line)
                        self.assertNotIn((dataset["repo_id"], row["row_index"]), locators)
                        self.assertNotIn(row["content_sha256"], content)
                        locators.add((dataset["repo_id"], row["row_index"]))
                        content.add(row["content_sha256"])
                        self.assertEqual(
                            hashlib.sha256(row["text"].encode()).hexdigest(),
                            row["content_sha256"],
                        )

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                module.collect_sampled_rows(
                    output,
                    source,
                    minimum_utf8_bytes={contract.repo_id: 96 for contract in module.DATASETS},
                    minimum_rows={contract.repo_id: 2 for contract in module.DATASETS},
                )

    def test_download_ceiling_applies_to_whole_collection(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = FixtureSource(module, root)
            first_two_bytes = sum(
                source.shards[contract.repo_id][0].bytes
                for contract in module.DATASETS[:2]
            )
            with self.assertRaisesRegex(ValueError, "campaign download ceiling"):
                module.collect_sampled_rows(
                    root / "sampled",
                    source,
                    minimum_utf8_bytes={
                        contract.repo_id: 96 for contract in module.DATASETS
                    },
                    minimum_rows={
                        contract.repo_id: 2 for contract in module.DATASETS
                    },
                    max_download_bytes=first_two_bytes,
                )

    def test_preflights_every_source_before_any_download(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = DeniedFinalSource(module, root)
            with self.assertRaisesRegex(PermissionError, "gated source access denied"):
                module.collect_sampled_rows(
                    root / "sampled",
                    source,
                    minimum_utf8_bytes={
                        contract.repo_id: 96 for contract in module.DATASETS
                    },
                    minimum_rows={
                        contract.repo_id: 2 for contract in module.DATASETS
                    },
                )
            self.assertEqual(
                source.preflighted,
                [contract.repo_id for contract in module.DATASETS],
            )
            self.assertEqual(source.materialized, [])
            self.assertFalse((root / "sampled").exists())

    def test_publish_never_replaces_racing_output_directory(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "sampled"
            source = RacingOutputSource(module, root, output)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                module.collect_sampled_rows(
                    output,
                    source,
                    minimum_utf8_bytes={
                        contract.repo_id: 96 for contract in module.DATASETS
                    },
                    minimum_rows={
                        contract.repo_id: 2 for contract in module.DATASETS
                    },
                )
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])

    def test_retained_source_descriptor_rejects_in_place_mutation(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = FixtureSource(module, root)
            contract = module.DATASETS[0]
            shard = source.shards[contract.repo_id][0]
            path = source.paths[(contract.repo_id, shard.path)]
            with self.assertRaisesRegex(ValueError, "changed during collection"):
                with module._verified_source_stream(path, shard) as stream:
                    self.assertTrue(stream.read(1))
                    with path.open("r+b") as writer:
                        first = writer.read(1)
                        writer.seek(0)
                        writer.write(bytes([first[0] ^ 0x01]))

    def test_repeated_collection_is_byte_identical(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_root = root / "source"
            source_root.mkdir()
            source = FixtureSource(module, source_root)
            targets = {contract.repo_id: 96 for contract in module.DATASETS}
            row_targets = {contract.repo_id: 2 for contract in module.DATASETS}
            first = root / "first"
            second = root / "second"
            module.collect_sampled_rows(
                first,
                source,
                minimum_utf8_bytes=targets,
                minimum_rows=row_targets,
            )
            module.collect_sampled_rows(
                second,
                source,
                minimum_utf8_bytes=targets,
                minimum_rows=row_targets,
            )
            first_files = sorted(
                path.relative_to(first) for path in first.rglob("*") if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second) for path in second.rglob("*") if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())

    def test_duplicate_only_source_fails_without_partial_output(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = FixtureSource(module, root, duplicate_only=True)
            output = root / "sampled"
            with self.assertRaisesRegex(ValueError, "row/byte targets"):
                module.collect_sampled_rows(
                    output,
                    source,
                    minimum_utf8_bytes={
                        contract.repo_id: 1 for contract in module.DATASETS
                    },
                    minimum_rows={
                        contract.repo_id: 2 for contract in module.DATASETS
                    },
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
