import copy
import hashlib
import itertools
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "qualify-stage7-recipe-freeze.py"
)
canonical = MODULE["canonical"]
recipe_grid = MODULE["recipe_grid"]
qualify = MODULE["qualify"]
QUALIFIER_GLOBALS = qualify.__globals__
Stage7Error = MODULE["Stage7Error"]
RATES = MODULE["RATES"]
RATE_PROFILE = MODULE["RATE_PROFILE"]
TASKS = MODULE["TASKS"]

PACKAGE_FIXTURES = {
    "D2": (
        "trp1_d28b0b2bc13902e113c33f83617e86a5a339356aee4b36ef432beeff41c05d19",
        "54534c5432504b47010001000100000068000000000000000100000001000000"
        "0100000000000000010000000000000001000000000000000200000000000000"
        "0000000000000000000000000000000000000000000000007701000000000000"
        "0055000000000000",
    ),
    "B3": (
        "trp1_1f702f6bfef290ccf1aedc8514f25ce1c37c23854b8b17bf73769ebb8567fb25",
        "54534c5432504b47010002000100000068000000000000000100000001000000"
        "0100000000000000010000000000000001000000000000000200000000000000"
        "0000000000000000000000000000000000000000000000007701000000000000"
        "0079000000000000",
    ),
    "S34": (
        "trp1_9c8c748ef8c283335b75d84506aec328e43b1a3335a0c225e1e455cfb6fe8011",
        "54534c5432504b47010003000100000068000000000000000100000001000000"
        "0100000000000000010000000000000001000000000000000200000000000000"
        "0000000000000000000000000000000000000000000000007701000000000000"
        "0000000000000000",
    ),
}


def salt_package(codec):
    package_id, encoded = PACKAGE_FIXTURES[codec]
    return bytes.fromhex(encoded), package_id


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def member_digests(label, count):
    return [
        "sha256:" + hashlib.sha256(f"{label}:{ordinal}".encode()).hexdigest()
        for ordinal in range(count)
    ]


def partition_provenance(label, seed, tokenizer_digest):
    members = member_digests(label, 512)
    partition = {
        "members": members,
        "datasets": [
            {
                "repo_id": "allenai/c4",
                "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
                "fraction_ppm": 500_000,
            },
            {
                "repo_id": "open-web-math/open-web-math",
                "revision": "fde8ef8de2300f5e778f56261843dab89f230815",
                "fraction_ppm": 250_000,
            },
            {
                "repo_id": "bigcode/starcoderdata",
                "revision": "9fc30b578cedaec69e47302df72cf00feed7c8c4",
                "fraction_ppm": 250_000,
            },
        ],
        "sampling_seed": seed,
        "tokenizer_digest": tokenizer_digest,
        "ordered_token_digest": digest(members),
        "sequence_count": 512,
        "tokens_per_sequence": 2_048,
    }
    partition["id"] = digest(partition)
    return partition


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def file_record(root: Path, name: str, payload: bytes):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def source_repo(root: Path):
    repo = root / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README").write_text("stage7 source\n", encoding="utf-8")
    qualifier = repo / "scripts/qualify-stage7-recipe-freeze.py"
    qualifier.parent.mkdir()
    qualifier.write_bytes(
        Path(MODULE["__file__"]).read_bytes()
        if "__file__" in MODULE
        else (Path(__file__).resolve().parents[1] / "qualify-stage7-recipe-freeze.py").read_bytes()
    )
    source_vectors = (
        Path(__file__).resolve().parents[2]
        / "crates/tritium-spec/data/training/v3/vectors/v3.json"
    )
    vectors = repo / "crates/tritium-spec/data/training/v3/vectors/v3.json"
    vectors.parent.mkdir(parents=True)
    vectors.write_bytes(source_vectors.read_bytes())
    subprocess.run(
        ["git", "-C", str(repo), "add", "README", "scripts", "crates"],
        check=True,
    )
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "source",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    QUALIFIER_GLOBALS["__file__"] = str(qualifier)
    QUALIFIER_GLOBALS["_native_source_identity"] = (
        lambda revision=revision: f"source-git:{revision}"
    )
    return repo, revision


def safetensors_bytes():
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32", "shape": [100, 100], "data_offsets": [0, 40_000]
        },
        "model.norm.weight": {
            "dtype": "F32", "shape": [100], "data_offsets": [40_000, 40_400]
        },
    }
    header = canonical(tensors)
    return len(header).to_bytes(8, "little") + header + bytes(40_400)


def model_file_records(root: Path, weights: bytes):
    payloads = {
        ".gitattributes": b"lfs\n",
        "README.md": b"model\n",
        "config.json": b'{"vocab_size":4096}\n',
        "generation_config.json": b"{}\n",
        "merges.txt": b"#version\n",
        "model.safetensors": weights,
        "special_tokens_map.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "vocab.json": b"{}\n",
    }
    return [file_record(root, name, payloads[name]) for name in sorted(payloads)]


def tokenizer_identity(records):
    tokenizer_files = {
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    selected = [record for record in records if record["path"] in tokenizer_files]
    return digest(selected)


def write_token_evidence_pack(root: Path, provenance, tokenizer_digest):
    tokens = bytearray()
    partitions = {}
    global_ordinal = 0
    dataset_contracts = {
        "allenai/c4": ("en", None, "text"),
        "open-web-math/open-web-math": ("default", None, "text"),
        "bigcode/starcoderdata": ("default", "python", "content"),
    }
    for partition_ordinal, (name, campaign_partition) in enumerate(
        provenance.items()
    ):
        sequences = []
        for ordinal in range(512):
            dataset_ordinal = 0 if ordinal < 256 else 1 if ordinal < 384 else 2
            dataset = campaign_partition["datasets"][dataset_ordinal]
            dataset_config, dataset_data_dir, text_field = dataset_contracts[
                dataset["repo_id"]
            ]
            row_index = partition_ordinal * 10_000 + ordinal
            sequence_tokens = (
                global_ordinal.to_bytes(4, "little") + bytes((2_048 - 1) * 4)
            )
            token_sha = hashlib.sha256(sequence_tokens).hexdigest()
            sequence = {
                "dataset_repo_id": dataset["repo_id"],
                "dataset_revision": dataset["revision"],
                "dataset_config": dataset_config,
                "dataset_data_dir": dataset_data_dir,
                "dataset_split": "train",
                "source_rows": [
                    {
                        "row_index": row_index,
                        "text_field": text_field,
                        "content_sha256": hashlib.sha256(
                            f"{name}:{dataset['repo_id']}:{row_index}".encode()
                        ).hexdigest(),
                    }
                ],
                "token_offset": global_ordinal * 2_048,
                "token_count": 2_048,
                "token_sha256": "sha256:" + token_sha,
            }
            sequence["id"] = digest(sequence)
            sequences.append(sequence)
            tokens.extend(sequence_tokens)
            global_ordinal += 1
        campaign_partition["members"] = [sequence["id"] for sequence in sequences]
        campaign_partition["ordered_token_digest"] = digest(
            campaign_partition["members"]
        )
        campaign_partition["id"] = digest(
            {
                key: value
                for key, value in campaign_partition.items()
                if key != "id"
            }
        )
        partitions[name] = {
            "sampling_seed": campaign_partition["sampling_seed"],
            "sequences": sequences,
        }
    token_record = file_record(root, "tokens/stage7.u32le", bytes(tokens))
    token_record["path"] = "stage7.u32le"
    manifest = {
        "schema": "tritium.stage7-token-evidence-pack.v1",
        "tokenizer_digest": tokenizer_digest,
        "tokenizer_vocab_size": 4_096,
        "token_encoding": "u32le",
        "tokens": token_record,
        "partitions": partitions,
    }
    manifest["pack_id"] = digest(manifest)
    manifest_path = root / "tokens/manifest.json"
    write_json(manifest_path, manifest)
    return file_record(root, "tokens/manifest.json", manifest_path.read_bytes())


def matched_control(candidate_id, grid):
    recipe = copy.deepcopy(grid[candidate_id])
    recipe["curvature"] = "input-hessian"
    return digest(recipe)


def candidate_for(grid, rate, codec):
    profile = RATE_PROFILE[rate]
    for candidate_id, recipe in grid.items():
        if (
            recipe["profile"] == profile
            and recipe["group_size"] == 128
            and recipe["codec"] == codec
            and recipe["max_planes"] == (2 if rate == "R4" else 3)
            and recipe["rotation"] == {"kind": "signed-rht", "seed": 0}
            and recipe["curvature"] == "guided-fisher"
            and recipe["solver"]["variant"] == "joint+feedback+output-recon"
        ):
            return candidate_id
    raise AssertionError(f"missing target {rate}")


def refinement_row(
    root: Path,
    *,
    mode,
    parent,
    rate,
    soft_method,
    cap,
    parent_ppl,
    final_ppl,
    refinement_corpus_id,
    validation_id,
    soft_policy,
    codec,
    release,
    source_revision,
):
    checkpoints = []
    schedule = [cap // 8, cap // 4, cap // 2, cap]
    for ordinal, tokens in enumerate(schedule):
        ppl = final_ppl + (3 - ordinal) * 0.2
        artifact_payload, package_id = salt_package(codec)
        artifact = file_record(
            root,
            f"refinement/{mode}-{soft_method or rate}-{tokens}.tsalt2",
            artifact_payload,
        )
        evaluation = {
            "schema": "tritium.stage7-refinement-evaluation.v1",
            "result": "pass",
            "release": release,
            "source_revision": source_revision,
            "parent_candidate_id": parent,
            "mode": mode,
            "soft_method": soft_method,
            "refinement_corpus_id": refinement_corpus_id,
            "validation_id": validation_id,
            "tokens": tokens,
            "artifact_sha256": artifact["sha256"],
            "package_id": package_id,
            "validation_ppl": ppl,
            "teacher_kl": 0.4 - ordinal * 0.05,
        }
        evaluation["evaluation_id"] = digest(evaluation)
        evaluation_path = (
            f"refinement/evaluation-{mode}-{soft_method or rate}-{tokens}.json"
        )
        evaluation_record = file_record(
            root, evaluation_path, canonical(evaluation) + b"\n"
        )
        checkpoints.append(
            {
                "tokens": tokens,
                "validation_ppl": ppl,
                "teacher_kl": 0.4 - ordinal * 0.05,
                "artifact": artifact,
                "package_id": package_id,
                "codec": codec.lower(),
                "serialized_bytes": len(artifact_payload),
                "resident_bytes": 3,
                "tensor_count": 1,
                "trits_changed": mode == "short-pv",
                "hard_reload_max_abs_error": 0.0,
                "hard_reload_tolerance": 1e-4,
                "evaluation_receipt": evaluation_record,
            }
        )
    row = {
        "refinement_id": "pending",
        "mode": mode,
        "parent_candidate_id": parent,
        "rate": rate,
        "soft_method": soft_method,
        "soft_policy": soft_policy,
        "refinement_corpus_id": refinement_corpus_id,
        "validation_id": validation_id,
        "parent_validation_ppl": parent_ppl,
        "checkpoints": checkpoints,
    }
    row["refinement_id"] = digest(
        {key: value for key, value in row.items() if key != "refinement_id"}
    )
    return row


def reseal_refinement(root: Path, refinement):
    for checkpoint in refinement["checkpoints"]:
        evaluation_path = root / checkpoint["evaluation_receipt"]["path"]
        evaluation = json.loads(evaluation_path.read_bytes())
        evaluation["validation_ppl"] = checkpoint["validation_ppl"]
        evaluation["teacher_kl"] = checkpoint["teacher_kl"]
        evaluation["evaluation_id"] = digest(
            {
                key: value
                for key, value in evaluation.items()
                if key != "evaluation_id"
            }
        )
        write_json(evaluation_path, evaluation)
        checkpoint["evaluation_receipt"].update(
            file_record(
                root,
                checkpoint["evaluation_receipt"]["path"],
                evaluation_path.read_bytes(),
            )
        )
    refinement["refinement_id"] = digest(
        {key: value for key, value in refinement.items() if key != "refinement_id"}
    )


def fixture(root: Path):
    source, revision = source_repo(root)
    model_root = root / "model"
    model_root.mkdir()
    model_files = model_file_records(model_root, safetensors_bytes())
    model_revision = "effd688a12921b4cc83e3312b6feb579f70f9c71"
    model = {
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B",
        "revision": model_revision,
        "files": model_files,
    }
    model_id = digest(model)
    QUALIFIER_GLOBALS["SMOLLM2_17B_MODEL_ID"] = model_id
    smoke_model_root = root / "smoke-model"
    smoke_model_root.mkdir()
    smoke_model = {
        "repo_id": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "files": model_file_records(smoke_model_root, safetensors_bytes()),
    }
    smoke_model_id = digest(smoke_model)
    QUALIFIER_GLOBALS["SMOLLM2_135M_MODEL_ID"] = smoke_model_id
    tokenizer_digest = tokenizer_identity(model_files)
    assert tokenizer_digest == tokenizer_identity(smoke_model["files"])
    provenance = {
        "calibration": partition_provenance("calibration", 11, tokenizer_digest),
        "refinement": partition_provenance("refinement", 12, tokenizer_digest),
        "validation": partition_provenance("validation", 13, tokenizer_digest),
        "evaluation": partition_provenance("evaluation", 14, tokenizer_digest),
    }
    token_evidence_pack = write_token_evidence_pack(
        root, provenance, tokenizer_digest
    )
    calibration_id = provenance["calibration"]["id"]
    refinement_corpus_id = provenance["refinement"]["id"]
    validation_id = provenance["validation"]["id"]
    evaluation_id = provenance["evaluation"]["id"]
    grid = recipe_grid(
        source_model_id=model_id,
        calibration_id=calibration_id,
        evaluation_id=evaluation_id,
        quantized_parameters=10_000,
        preserved_bytes=400,
    )
    release = "1.1.0-rc.0"

    smoke_payload, smoke_package_id = salt_package("B3")
    smoke_artifact = file_record(root, "smoke/model.tsalt2", smoke_payload)
    smoke_evaluation_members = provenance["calibration"]["members"][:128]
    smoke_evaluation_id = digest(smoke_evaluation_members)
    smoke_execution = {
        "schema": "tritium.stage7-smoke-execution.v1",
        "result": "pass",
        "release": release,
        "source_revision": revision,
        "model_id": smoke_model_id,
        "model_revision": smoke_model["revision"],
        "evaluation_id": smoke_evaluation_id,
        "artifact_sha256": smoke_artifact["sha256"],
        "stages": [
            {"name": name, "result": "pass"}
            for name in ("capture", "fit", "allocate", "package", "evaluate")
        ],
    }
    smoke_execution_path = root / "smoke/execution.json"
    write_json(smoke_execution_path, smoke_execution)
    smoke_log = file_record(
        root,
        smoke_execution_path.relative_to(root).as_posix(),
        smoke_execution_path.read_bytes(),
    )
    smoke = {
        "schema": "tritium.stage7-smoke.v1",
        "result": "pass",
        "release": release,
        "source_revision": revision,
        "model_id": smoke_model_id,
        "model_revision": smoke_model["revision"],
        "evaluation_id": smoke_evaluation_id,
        "artifact": smoke_artifact,
        "package_id": smoke_package_id,
        "codec": "b3",
        "serialized_bytes": len(smoke_payload),
        "resident_bytes": 3,
        "tensor_count": 1,
        "execution_log": smoke_log,
    }
    smoke_path = root / "smoke-receipt.json"
    write_json(smoke_path, smoke)

    sanitizer = file_record(root, "native/compute-sanitizer.log", b"ERROR SUMMARY: 0 errors\n")
    native_cases = []
    packing = {"D2": "direct-2bit", "B3": "radix-3", "S34": "s34"}
    for codec, group, planes, mode in itertools.product(
        ("D2", "B3", "S34"), (64, 128, 256), (2, 3), ("exact", "fast")
    ):
        schedules = ("uniform", "mixed")
        for schedule, shape in itertools.product(schedules, ("aligned", "short")):
            native_cases.append(
                {
                    "codec": codec,
                    "group_size": group,
                    "planes": planes,
                    "mode": mode,
                    "rows": 64 if shape == "aligned" else 3,
                    "columns": group * 2 if shape == "aligned" else group + 17,
                    "short_final_group": shape == "short",
                    "plane_schedule": schedule,
                    "packing": packing[codec],
                    "cpu_max_abs_error": 0.0,
                    "cuda_max_abs_error": 0.0,
                    "tolerance": 1e-4,
                    "dense_materialized_bytes": 0,
                }
            )
    native = {
        "schema": "tritium.stage7-native-kernels.v1",
        "result": "pass",
        "release": release,
        "source_revision": revision,
        "model_revision": model_revision,
        "physical_device": "cuda:0:GPU-1",
        "driver": "999.1",
        "sanitizer_version": "Compute Sanitizer 999.1",
        "sanitizer_log": sanitizer,
        "cases": native_cases,
    }
    native_path = root / "native-receipt.json"
    write_json(native_path, native)

    hestia_vector_digest = (
        "sha256:9ab8b18c122e2a3721663894744b766f"
        "a213057193c9dcc5ac64f1b362acf20a"
    )
    hestia_gate = {
        "schema": "tritium.stage7-hestia-gate-c.v1",
        "result": "pass",
        "release": release,
        "source_revision": revision,
        "gradcheck": {
            "suite": "tritium-train/gradcheck_hestia",
            "result": "pass",
            "inputs": ["weight", "temperature"],
            "max_relative_error": 1e-5,
            "tolerance": 2e-3,
        },
        "portable_cpu": {
            "backend": "cpu",
            "result": "pass",
            "manifest_version": 3,
            "operation": "graph.hestia_relax",
            "vector_digest": hestia_vector_digest,
            "case_count": 5,
            "physical_device": "cpu",
            "driver": "rust-reference",
        },
        "portable_cuda": {
            "backend": "cuda",
            "result": "pass",
            "manifest_version": 3,
            "operation": "graph.hestia_relax",
            "vector_digest": hestia_vector_digest,
            "case_count": 5,
            "physical_device": "cuda:0:GPU-1",
            "driver": "999.1",
        },
    }
    hestia_gate_path = root / "hestia-gate-c.json"
    write_json(hestia_gate_path, hestia_gate)

    campaign = {
        "schema": "tritium.stage7-campaign.v1",
        "release": release,
        "source_revision": revision,
        "run_id": "stage7-run-1",
        "model": model,
        "smoke_model": smoke_model,
        "smoke_provenance": {
            "evaluation_id": smoke_evaluation_id,
            "evaluation_members": smoke_evaluation_members,
            "calibration_id": calibration_id,
            "dataset_repo_id": "allenai/c4",
            "dataset_revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            "sampling_seed": 11,
            "tokenizer_digest": tokenizer_digest,
            "ordered_token_digest": smoke_evaluation_id,
            "sequence_count": 128,
            "tokens_per_sequence": 2_048,
            "prefix_start": 0,
            "prefix_end": 128,
        },
        "provenance": provenance,
        "thresholds": {
            "r3_gap_closure_min": 0.25,
            "metadata_bpw_max": 0.01,
            "scale_only_token_cap": 8_000_000,
            "short_pv_token_cap": 32_000_000,
        },
        "recipe_count": 1404,
        "recipe_grid_id": digest(sorted(grid)),
        "token_evidence_pack": token_evidence_pack,
        "evidence": [
            {"kind": "smoke", **file_record(root, smoke_path.name, smoke_path.read_bytes())},
            {
                "kind": "native-kernels",
                **file_record(root, native_path.name, native_path.read_bytes()),
            },
            {
                "kind": "hestia-gate-c",
                **file_record(
                    root, hestia_gate_path.name, hestia_gate_path.read_bytes()
                ),
            },
        ],
    }
    campaign_path = root / "campaign.json"
    write_json(campaign_path, campaign)

    codec_targets = {
        (rate, codec): candidate_for(grid, rate, codec)
        for rate in RATES
        for codec in (("D2",) if rate == "R4" else ("D2", "B3", "S34"))
    }
    codec_controls = {
        identity: matched_control(candidate_id, grid)
        for identity, candidate_id in codec_targets.items()
    }
    targets = {
        "R2": codec_targets[("R2", "B3")],
        "R3": codec_targets[("R3", "B3")],
        "R4": codec_targets[("R4", "D2")],
    }
    controls = {
        rate: matched_control(targets[rate], grid) for rate in RATES
    }
    task_metrics = {task: 0.60 for task in TASKS}

    def measurement(candidate_id, stage, ordinal):
        recipe = grid[candidate_id]
        rate = next(rate for rate in RATES if recipe["profile"] == RATE_PROFILE[rate])
        rate_ordinal = RATES.index(rate)
        codec = recipe["codec"]
        is_target = candidate_id == codec_targets.get((rate, codec))
        is_control = candidate_id == codec_controls.get((rate, codec))
        codec_penalty = {"B3": 0.0, "S34": 0.02, "D2": 0.04}[codec]
        output_loss = (
            0.8 - rate_ordinal * 0.1 + codec_penalty
            if is_target
            else 0.85 - rate_ordinal * 0.1 + codec_penalty
            if is_control
            else 100.0 + ordinal / 10_000
        )
        row = {
            "candidate_id": candidate_id,
            "track": "ptq",
            "physical_bytes": 104,
            "resident_bytes": 403,
            "output_loss": output_loss,
            "heldout_ppl": None,
            "task_metrics": {},
            "runtime_ms": 10.0 + rate_ordinal if is_target or is_control else 100.0,
            "artifact": None,
            "physical_report": None,
            "correct": True,
        }
        if stage != "full-model":
            return row
        row["heldout_ppl"] = (
            14.0 - rate_ordinal + {"B3": 0.0, "S34": 0.25, "D2": 0.5}[codec]
            if is_target
            else 15.0 - rate_ordinal + {"B3": 0.0, "S34": 0.25, "D2": 0.5}[codec]
        )
        row["task_metrics"] = dict(task_metrics)
        artifact_payload, package_id = salt_package(codec)
        artifact = file_record(root, f"full/{ordinal:04d}.tsalt2", artifact_payload)
        report = {
            "schema": "tritium.stage7-physical-report.v1",
            "result": "pass",
            "release": release,
            "source_revision": revision,
            "recipe_id": candidate_id,
            "artifact_sha256": artifact["sha256"],
            "package_id": package_id,
            "codec": codec.lower(),
            "tensor_count": 1,
            "package_resident_bytes": 3,
            "quantized_parameter_count": 10_000,
            "components": {
                "trit_payload": 1,
                "scales": 2,
                "allocation_map": 0,
                "transform": 24,
                "padding": 4,
                "tensor_headers": 49,
                "container": 24,
                "preserved_tensors": 400,
            },
            "matrix_bytes": row["physical_bytes"],
            "artifact_bytes": row["physical_bytes"] + 400,
            "steady_resident_bytes": row["resident_bytes"],
            "peak_resident_bytes": row["resident_bytes"] + 100,
            "dense_materialized_bytes": 0,
        }
        report_path = root / f"full/{ordinal:04d}-physical.json"
        write_json(report_path, report)
        row["artifact"] = artifact
        row["physical_report"] = file_record(
            root, report_path.relative_to(root).as_posix(), report_path.read_bytes()
        )
        return row

    all_ids = sorted(grid)
    promoted = sorted(set(codec_targets.values()) | set(codec_controls.values()))
    first = {
        "name": "one-layer",
        "input_ids": all_ids,
        "measurements": [
            measurement(candidate_id, "one-layer", ordinal)
            for ordinal, candidate_id in enumerate(all_ids)
        ],
        "promoted_ids": promoted,
    }
    second = {
        "name": "four-layer",
        "input_ids": promoted,
        "measurements": [
            measurement(candidate_id, "four-layer", ordinal)
            for ordinal, candidate_id in enumerate(promoted)
        ],
        "promoted_ids": promoted,
    }
    full = {
        "name": "full-model",
        "input_ids": promoted,
        "measurements": [
            measurement(candidate_id, "full-model", ordinal)
            for ordinal, candidate_id in enumerate(promoted)
        ],
        "promoted_ids": sorted(codec_targets.values()),
    }

    scale = [
        refinement_row(
            root,
            mode="scale-only",
            parent=targets[rate],
            rate=rate,
            soft_method=None,
            cap=8_000_000,
            parent_ppl=15.0 - ordinal,
            final_ppl=12.0 - ordinal,
            refinement_corpus_id=refinement_corpus_id,
            validation_id=validation_id,
            soft_policy={"kind": "none"},
            codec=grid[targets[rate]]["codec"],
            release=release,
            source_revision=revision,
        )
        for ordinal, rate in enumerate(RATES)
    ]
    s2kf_record_digests = {
        "model.layers.0.self_attn.q_proj.weight": "blake3:" + "f" * 64,
    }
    sensitivity = {
        "schema": "tritium.stage7-s2kf-sensitivity.v1",
        "result": "pass",
        "release": release,
        "source_revision": revision,
        "model_id": model_id,
        "calibration_id": calibration_id,
        "parent_candidate_id": targets["R3"],
        "sensitivity_method": (
            "standardized-sigmoid(input-gram-trace*output-fisher-mean)"
        ),
        "s2kf_source_model_digest": model_id,
        "s2kf_activation_cache_digest": "sha256:" + "9" * 64,
        "s2kf_token_stream_digest": provenance["calibration"][
            "ordered_token_digest"
        ],
        "s2kf_record_digests": s2kf_record_digests,
        "s2kf_manifest_id": digest(s2kf_record_digests),
        "tensor_scores": {
            "model.layers.0.self_attn.q_proj.weight": 1.0,
        },
    }
    sensitivity["evidence_id"] = digest(sensitivity)
    sensitivity_path = root / "refinement/hestia-sensitivity.json"
    write_json(sensitivity_path, sensitivity)
    sensitivity_record = file_record(
        root,
        sensitivity_path.relative_to(root).as_posix(),
        sensitivity_path.read_bytes(),
    )

    def soft_policy(method):
        if method == "ste-soft":
            return {
                "kind": "ste-soft",
                "hard_boundary_fraction": 0.8,
                "hard_export": "hard-trits-scale-only",
            }
        return {
            "kind": "hestia-relaxation",
            "tau_initial": 1.0,
            "tau_floor": 0.01,
            "schedule": "exponential",
            "total_tokens": 32_000_000,
            "sensitivity_alpha": 1.0,
            "sensitivity_evidence": sensitivity_record,
            "floor_reached_by_fraction": 0.8,
            "hard_boundary_fraction": 0.8,
            "hard_export": "hard-trits-scale-only",
        }

    pv = [
        refinement_row(
            root,
            mode="short-pv",
            parent=targets["R3"],
            rate="R3",
            soft_method=method,
            cap=32_000_000,
            parent_ppl=11.0,
            final_ppl=9.0 if method == "ste-soft" else 8.5,
            refinement_corpus_id=refinement_corpus_id,
            validation_id=validation_id,
            soft_policy=soft_policy(method),
            codec=grid[targets["R3"]]["codec"],
            release=release,
            source_revision=revision,
        )
        for method in ("ste-soft", "hestia-relaxation")
    ]
    salt_baselines = []
    for ordinal, rate in enumerate(RATES):
        codec = "d2" if rate == "R4" else "b3"
        baseline_id = digest(
            {
                "method": "salt-v1",
                "rate": rate,
                "codec": codec,
                "source_model_id": model_id,
                "evaluation_id": evaluation_id,
            }
        )
        physical = 104
        artifact_payload, package_id = salt_package(codec.upper())
        artifact = file_record(
            root, f"baselines/salt-v1-{rate}.tsalt2", artifact_payload
        )
        report = {
            "schema": "tritium.stage7-physical-report.v1",
            "result": "pass",
            "release": release,
            "source_revision": revision,
            "recipe_id": baseline_id,
            "artifact_sha256": artifact["sha256"],
            "package_id": package_id,
            "codec": codec,
            "tensor_count": 1,
            "package_resident_bytes": 3,
            "quantized_parameter_count": 10_000,
            "components": {
                "trit_payload": 1,
                "scales": 2,
                "allocation_map": 0,
                "transform": 24,
                "padding": 4,
                "tensor_headers": 49,
                "container": 24,
                "preserved_tensors": 400,
            },
            "matrix_bytes": physical,
            "artifact_bytes": physical + 400,
            "steady_resident_bytes": 403,
            "peak_resident_bytes": 503,
            "dense_materialized_bytes": 0,
        }
        report_path = root / f"baselines/salt-v1-{rate}-physical.json"
        write_json(report_path, report)
        salt_baselines.append(
            {
                "rate": rate,
                "codec": codec,
                "baseline_id": baseline_id,
                "physical_bytes": physical,
                "resident_bytes": 403,
                "heldout_ppl": 20.0 - ordinal,
                "task_metrics": dict(task_metrics),
                "artifact": artifact,
                "physical_report": file_record(
                    root,
                    report_path.relative_to(root).as_posix(),
                    report_path.read_bytes(),
                ),
            }
        )
    trace = {
        "schema": "tritium.stage7-execution.v1",
        "release": release,
        "source_revision": revision,
        "run_id": "stage7-run-1",
        "campaign_sha256": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
        "stages": [first, second, full],
        "baselines": {
            "bf16": {
                "heldout_ppl": 10.0,
                "task_metrics": {task: 0.70 for task in TASKS},
            },
            "salt_v1": salt_baselines,
        },
        "refinements": scale + pv,
    }
    trace_path = root / "trace.json"
    write_json(trace_path, trace)
    return {
        "source": source,
        "revision": revision,
        "model_root": model_root,
        "smoke_model_root": smoke_model_root,
        "campaign_path": campaign_path,
        "trace_path": trace_path,
        "campaign": campaign,
        "trace": trace,
        "grid": grid,
        "targets": targets,
        "controls": controls,
        "codec_targets": codec_targets,
        "codec_controls": codec_controls,
        "pv": pv,
    }


class Stage7RecipeFreezeTests(unittest.TestCase):
    def test_rejects_campaign_without_token_evidence_pack(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            del state["campaign"]["token_evidence_pack"]
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "token evidence pack"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_resealed_pack_with_changed_sequence_tokens(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = fixture(root)
            manifest_path = root / state["campaign"]["token_evidence_pack"]["path"]
            manifest = json.loads(manifest_path.read_bytes())
            tokens_path = manifest_path.parent / manifest["tokens"]["path"]
            tokens = bytearray(tokens_path.read_bytes())
            tokens[0] ^= 1
            tokens_path.write_bytes(tokens)
            manifest["tokens"] = file_record(
                manifest_path.parent,
                manifest["tokens"]["path"],
                tokens_path.read_bytes(),
            )
            manifest["pack_id"] = digest(
                {key: value for key, value in manifest.items() if key != "pack_id"}
            )
            write_json(manifest_path, manifest)
            state["campaign"]["token_evidence_pack"] = file_record(
                root,
                state["campaign"]["token_evidence_pack"]["path"],
                manifest_path.read_bytes(),
            )
            write_json(state["campaign_path"], state["campaign"])
            state["trace"]["campaign_sha256"] = hashlib.sha256(
                state["campaign_path"].read_bytes()
            ).hexdigest()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "token payload digest"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_oversized_token_payload_before_record_open(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = fixture(root)
            manifest_path = root / state["campaign"]["token_evidence_pack"]["path"]
            manifest = json.loads(manifest_path.read_bytes())
            tokens_path = manifest_path.parent / manifest["tokens"]["path"]
            tokens_path.write_bytes(tokens_path.read_bytes() + b"\x00\x00\x00\x00")
            manifest["tokens"] = file_record(
                manifest_path.parent,
                manifest["tokens"]["path"],
                tokens_path.read_bytes(),
            )
            manifest["pack_id"] = digest(
                {key: value for key, value in manifest.items() if key != "pack_id"}
            )
            write_json(manifest_path, manifest)
            state["campaign"]["token_evidence_pack"] = file_record(
                root,
                state["campaign"]["token_evidence_pack"]["path"],
                manifest_path.read_bytes(),
            )
            write_json(state["campaign_path"], state["campaign"])
            state["trace"]["campaign_sha256"] = hashlib.sha256(
                state["campaign_path"].read_bytes()
            ).hexdigest()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(
                Stage7Error, "token evidence payload byte geometry differs"
            ):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_duplicate_token_payload_under_distinct_source_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = fixture(root)
            manifest_path = root / state["campaign"]["token_evidence_pack"]["path"]
            manifest = json.loads(manifest_path.read_bytes())
            sequences = manifest["partitions"]["calibration"]["sequences"]
            tokens_path = manifest_path.parent / manifest["tokens"]["path"]
            tokens = bytearray(tokens_path.read_bytes())
            width = 2_048 * 4
            tokens[width:2 * width] = tokens[:width]
            tokens_path.write_bytes(tokens)
            sequences[1]["token_sha256"] = sequences[0]["token_sha256"]
            sequences[1]["id"] = digest(
                {key: value for key, value in sequences[1].items() if key != "id"}
            )
            manifest["tokens"] = file_record(
                manifest_path.parent,
                manifest["tokens"]["path"],
                tokens_path.read_bytes(),
            )
            manifest["pack_id"] = digest(
                {key: value for key, value in manifest.items() if key != "pack_id"}
            )
            write_json(manifest_path, manifest)
            calibration = state["campaign"]["provenance"]["calibration"]
            calibration["members"][1] = sequences[1]["id"]
            calibration["ordered_token_digest"] = digest(calibration["members"])
            calibration["id"] = digest(
                {key: value for key, value in calibration.items() if key != "id"}
            )
            smoke = state["campaign"]["smoke_provenance"]
            smoke["evaluation_members"] = calibration["members"][:128]
            smoke["evaluation_id"] = digest(smoke["evaluation_members"])
            smoke["ordered_token_digest"] = smoke["evaluation_id"]
            smoke["calibration_id"] = calibration["id"]
            state["campaign"]["token_evidence_pack"] = file_record(
                root,
                state["campaign"]["token_evidence_pack"]["path"],
                manifest_path.read_bytes(),
            )
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "duplicate token sequences"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_duplicate_source_content_under_distinct_row_indices(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = fixture(root)
            manifest_path = root / state["campaign"]["token_evidence_pack"]["path"]
            manifest = json.loads(manifest_path.read_bytes())
            sequences = manifest["partitions"]["calibration"]["sequences"]
            sequences[1]["source_rows"][0]["content_sha256"] = sequences[0][
                "source_rows"
            ][0]["content_sha256"]
            sequences[1]["id"] = digest(
                {key: value for key, value in sequences[1].items() if key != "id"}
            )
            manifest["pack_id"] = digest(
                {key: value for key, value in manifest.items() if key != "pack_id"}
            )
            write_json(manifest_path, manifest)
            calibration = state["campaign"]["provenance"]["calibration"]
            calibration["members"][1] = sequences[1]["id"]
            calibration["ordered_token_digest"] = digest(calibration["members"])
            calibration["id"] = digest(
                {key: value for key, value in calibration.items() if key != "id"}
            )
            smoke = state["campaign"]["smoke_provenance"]
            smoke["evaluation_members"] = calibration["members"][:128]
            smoke["evaluation_id"] = digest(smoke["evaluation_members"])
            smoke["ordered_token_digest"] = smoke["evaluation_id"]
            smoke["calibration_id"] = calibration["id"]
            state["campaign"]["token_evidence_pack"] = file_record(
                root,
                state["campaign"]["token_evidence_pack"]["path"],
                manifest_path.read_bytes(),
            )
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "reuses source content"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_grid_is_complete_and_r4_is_control_only(self):
        grid = recipe_grid(
            source_model_id="sha256:" + "1" * 64,
            calibration_id="sha256:" + "2" * 64,
            evaluation_id="sha256:" + "3" * 64,
            quantized_parameters=10_000,
            preserved_bytes=400,
        )
        self.assertEqual(len(grid), 1404)
        counts = {
            profile: sum(recipe["profile"] == profile for recipe in grid.values())
            for profile in RATE_PROFILE.values()
        }
        self.assertEqual(counts, {"CompactV1": 648, "NearLosslessV1": 648, "R4Control": 108})
        self.assertTrue(all(
            recipe["codec"] == "D2" and recipe["max_planes"] == 2
            for recipe in grid.values() if recipe["profile"] == "R4Control"
        ))
        softened = next(
            recipe for recipe in grid.values()
            if recipe["solver"]["variant"] == "+softened-relay-basin"
        )
        self.assertEqual(
            softened["solver"]["relay_basins"],
            {
                "softened": True,
                "modulated": False,
                "steps": 12,
                "step_size": 0.05,
                "initial_sharpness": 30.0,
                "sharpness_multiplier": 2.0,
                "sharpness_interval": 4,
                "scale_bounds": [0.001, 8.0],
                "threshold_bounds": [0.05, 0.95],
                "shift_bounds": [-2.0, 2.0],
                "softened_threshold": 0.5,
            },
        )
        self.assertEqual(
            softened["solver"]["output_reconstruction"],
            {
                "enabled": True,
                "schedule": "sliding-windows",
                "block_count": 24,
                "window_size": 3,
                "stride": 1,
                "scale_refit_starts": 4,
                "fixed_trits": True,
            },
        )

    def test_qualifies_complete_terminal_freeze(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            output = Path(raw) / "qualification.json"
            result = qualify(
                state["campaign_path"],
                state["trace_path"],
                model_root=state["model_root"],
                smoke_model_root=state["smoke_model_root"],
                source_root=state["source"],
                output=output,
            )
            self.assertEqual(result["result"], "pass")
            self.assertEqual(
                result["frozen_ptq_recipe_ids"],
                {rate: state["targets"][rate] for rate in ("R2", "R3")},
            )
            self.assertEqual(result["r4_control_recipe_id"], state["targets"]["R4"])
            self.assertEqual(
                result["frozen_refined_recipe_id"], state["pv"][1]["refinement_id"]
            )
            expected_checkpoint = state["pv"][1]["checkpoints"][-1]
            self.assertEqual(
                result["frozen_refined_checkpoint"],
                {
                    "refinement_id": state["pv"][1]["refinement_id"],
                    "tokens": expected_checkpoint["tokens"],
                    "package_id": expected_checkpoint["package_id"],
                    "artifact_sha256": expected_checkpoint["artifact"]["sha256"],
                    "evaluation_sha256": expected_checkpoint["evaluation_receipt"][
                        "sha256"
                    ],
                },
            )
            self.assertTrue(all(
                state["codec_targets"][(rate, "D2")]
                in state["trace"]["stages"][2]["promoted_ids"]
                for rate in RATES
            ))
            self.assertEqual(json.loads(output.read_bytes()), result)

    def test_rejects_fake_bytes_under_official_hub_revision_label(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            synthetic_17b = QUALIFIER_GLOBALS["SMOLLM2_17B_MODEL_ID"]
            QUALIFIER_GLOBALS["SMOLLM2_17B_MODEL_ID"] = (
                "sha256:4be74d32a1a04f2984e9d118fdb165dd8cfbe972710796ab465a4c2152d58a08"
            )
            try:
                with self.assertRaisesRegex(Stage7Error, "frozen Hub revision"):
                    qualify(
                        state["campaign_path"], state["trace_path"],
                        model_root=state["model_root"],
                        smoke_model_root=state["smoke_model_root"],
                        source_root=state["source"],
                    )
            finally:
                QUALIFIER_GLOBALS["SMOLLM2_17B_MODEL_ID"] = synthetic_17b

    def test_rejects_incomplete_recipe_grid_binding(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["campaign"]["recipe_count"] = 9
            write_json(state["campaign_path"], state["campaign"])
            state["trace"]["campaign_sha256"] = hashlib.sha256(
                state["campaign_path"].read_bytes()
            ).hexdigest()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "complete frozen recipe grid"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_source_revision_not_equal_clean_head(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["campaign"]["source_revision"] = "f" * 40
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "clean repository HEAD"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_dirty_source_repository(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            (state["source"] / "untracked").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(Stage7Error, "must be clean"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_unrelated_repository_with_copied_qualifier(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            executing_source = QUALIFIER_GLOBALS["__file__"]
            unrelated_root = Path(raw) / "unrelated-root"
            unrelated_root.mkdir()
            unrelated, _ = source_repo(unrelated_root)
            QUALIFIER_GLOBALS["__file__"] = executing_source
            with self.assertRaisesRegex(Stage7Error, "executing qualifier repository"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=unrelated,
                )

    def test_rejects_overlap_across_four_data_partitions(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            provenance = state["campaign"]["provenance"]
            provenance["refinement"]["members"][0] = provenance["calibration"][
                "members"
            ][0]
            provenance["refinement"]["ordered_token_digest"] = digest(
                provenance["refinement"]["members"]
            )
            provenance["refinement"]["id"] = digest(
                {
                    key: value
                    for key, value in provenance["refinement"].items()
                    if key != "id"
                }
            )
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "four data partitions must be disjoint"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_smoke_members_not_equal_calibration_prefix(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            smoke = state["campaign"]["smoke_provenance"]
            smoke["evaluation_members"][0] = "sha256:" + "0" * 64
            smoke["evaluation_id"] = digest(smoke["evaluation_members"])
            smoke["ordered_token_digest"] = smoke["evaluation_id"]
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "calibration prefix"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_smoke_metadata_drift_from_parent_calibration(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["campaign"]["smoke_provenance"]["tokenizer_digest"] = (
                "sha256:" + "0" * 64
            )
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "parent calibration provenance"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_nonimmutable_dataset_revision_in_partition_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["campaign"]["provenance"]["calibration"]["datasets"][0][
                "revision"
            ] = "main"
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "dataset revision"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_other_commit_under_frozen_dataset_name(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["campaign"]["provenance"]["calibration"]["datasets"][0][
                "revision"
            ] = "f" * 40
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "frozen dataset revision"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_native_receipt_missing_one_shape(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            native_path = Path(raw) / "native-receipt.json"
            native = json.loads(native_path.read_bytes())
            native["cases"].pop()
            write_json(native_path, native)
            state["campaign"]["evidence"][1].update(
                file_record(Path(raw), native_path.name, native_path.read_bytes())
            )
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "case inventory"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_native_inventory_covers_mixed_schedule_at_two_plane_cap(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture(Path(raw))
            native = json.loads((Path(raw) / "native-receipt.json").read_bytes())
            mixed_p2 = [
                case for case in native["cases"]
                if case["planes"] == 2 and case["plane_schedule"] == "mixed"
            ]
            self.assertEqual(len(native["cases"]), 144)
            self.assertEqual(len(mixed_p2), 36)

    def test_records_terminal_negative_for_valid_native_kernel_gate_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            native_path = Path(raw) / "native-receipt.json"
            native = json.loads(native_path.read_bytes())
            native["result"] = "negative"
            native["cases"][0]["cuda_max_abs_error"] = 1.0
            write_json(native_path, native)
            state["campaign"]["evidence"][1].update(
                file_record(Path(raw), native_path.name, native_path.read_bytes())
            )
            write_json(state["campaign_path"], state["campaign"])
            state["trace"]["campaign_sha256"] = hashlib.sha256(
                state["campaign_path"].read_bytes()
            ).hexdigest()
            write_json(state["trace_path"], state["trace"])

            result = qualify(
                state["campaign_path"],
                state["trace_path"],
                model_root=state["model_root"],
                smoke_model_root=state["smoke_model_root"],
                source_root=state["source"],
            )

            self.assertEqual(result["result"], "negative")
            self.assertIn("native-kernel-gate-failed", result["freeze_reasons"])

    def test_rejects_missing_hestia_gate_c_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["campaign"]["evidence"].pop()
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "prerequisite evidence inventory"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hestia_gate_c_backends_using_different_vectors(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            gate_path = Path(raw) / "hestia-gate-c.json"
            gate = json.loads(gate_path.read_bytes())
            gate["portable_cuda"]["vector_digest"] = "sha256:" + "b" * 64
            write_json(gate_path, gate)
            state["campaign"]["evidence"][2].update(
                file_record(Path(raw), gate_path.name, gate_path.read_bytes())
            )
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "different vectors"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hestia_gate_c_noncanonical_v3_vectors(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            gate_path = Path(raw) / "hestia-gate-c.json"
            gate = json.loads(gate_path.read_bytes())
            fabricated = "sha256:" + "a" * 64
            gate["portable_cpu"]["vector_digest"] = fabricated
            gate["portable_cuda"]["vector_digest"] = fabricated
            write_json(gate_path, gate)
            state["campaign"]["evidence"][2].update(
                file_record(Path(raw), gate_path.name, gate_path.read_bytes())
            )
            write_json(state["campaign_path"], state["campaign"])
            state["trace"]["campaign_sha256"] = hashlib.sha256(
                state["campaign_path"].read_bytes()
            ).hexdigest()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "canonical V3 vectors"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hestia_gate_c_noncanonical_case_count(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            gate_path = Path(raw) / "hestia-gate-c.json"
            gate = json.loads(gate_path.read_bytes())
            gate["portable_cpu"]["case_count"] = 8
            gate["portable_cuda"]["case_count"] = 8
            write_json(gate_path, gate)
            state["campaign"]["evidence"][2].update(
                file_record(Path(raw), gate_path.name, gate_path.read_bytes())
            )
            write_json(state["campaign_path"], state["campaign"])
            state["trace"]["campaign_sha256"] = hashlib.sha256(
                state["campaign_path"].read_bytes()
            ).hexdigest()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "exactly five HESTIA cases"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_cherry_picked_promotion(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["trace"]["stages"][0]["promoted_ids"].pop()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "promotions differ"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_pass_report_that_contains_dense_shadow(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            row = state["trace"]["stages"][2]["measurements"][0]
            report_path = Path(raw) / row["physical_report"]["path"]
            report = json.loads(report_path.read_bytes())
            report["dense_materialized_bytes"] = 1
            write_json(report_path, report)
            row["physical_report"].update(
                file_record(Path(raw), row["physical_report"]["path"], report_path.read_bytes())
            )
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "result contradicts measured gates"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hash_consistent_non_salt_package(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            row = state["trace"]["stages"][2]["measurements"][0]
            artifact_path = Path(raw) / row["artifact"]["path"]
            artifact_path.write_bytes(b"x" * artifact_path.stat().st_size)
            row["artifact"].update(
                file_record(Path(raw), row["artifact"]["path"], artifact_path.read_bytes())
            )
            report_path = Path(raw) / row["physical_report"]["path"]
            report = json.loads(report_path.read_bytes())
            report["artifact_sha256"] = row["artifact"]["sha256"]
            write_json(report_path, report)
            row["physical_report"].update(
                file_record(
                    Path(raw), row["physical_report"]["path"], report_path.read_bytes()
                )
            )
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "semantic verification failed"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_native_package_verifier_from_other_source_revision(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            QUALIFIER_GLOBALS["_native_source_identity"] = lambda: (
                "source-git:" + "0" * 40
            )
            with self.assertRaisesRegex(Stage7Error, "native package verifier source"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_preserved_bytes_not_derived_from_source_tensors(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            row = state["trace"]["stages"][2]["measurements"][0]
            report_path = Path(raw) / row["physical_report"]["path"]
            report = json.loads(report_path.read_bytes())
            report["components"]["preserved_tensors"] = 399
            write_json(report_path, report)
            row["physical_report"].update(
                file_record(Path(raw), row["physical_report"]["path"], report_path.read_bytes())
            )
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "physical report totals"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_missing_hestia_ab_curve(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["trace"]["refinements"].pop()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "three scale curves and two PV"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_records_pv_ab_tradeoff_without_selecting_either_soft_method(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            hestia = state["trace"]["refinements"][-1]
            for ordinal, checkpoint in enumerate(hestia["checkpoints"]):
                checkpoint["teacher_kl"] = 0.60 - ordinal * 0.05
            reseal_refinement(Path(raw), hestia)
            write_json(state["trace_path"], state["trace"])
            result = qualify(
                state["campaign_path"], state["trace_path"],
                model_root=state["model_root"],
                smoke_model_root=state["smoke_model_root"],
                source_root=state["source"],
            )
            self.assertEqual(result["soft_method_ab"]["outcome"], "tradeoff")
            self.assertIsNone(result["soft_method_ab"]["winner"])
            self.assertEqual(
                result["frozen_refined_recipe_id"],
                state["trace"]["refinements"][1]["refinement_id"],
            )

    def test_rejects_unfrozen_hestia_temperature_policy(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            hestia = state["trace"]["refinements"][-1]
            hestia["soft_policy"]["tau_floor"] = 0.5
            hestia["refinement_id"] = digest(
                {key: value for key, value in hestia.items() if key != "refinement_id"}
            )
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "HESTIA soft policy differs"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hestia_sensitivity_outside_source_tensor_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            sensitivity_path = Path(raw) / "refinement/hestia-sensitivity.json"
            sensitivity = json.loads(sensitivity_path.read_bytes())
            sensitivity["tensor_scores"]["invented.weight"] = 0.5
            sensitivity["evidence_id"] = digest(
                {
                    key: value
                    for key, value in sensitivity.items()
                    if key != "evidence_id"
                }
            )
            write_json(sensitivity_path, sensitivity)
            hestia = state["trace"]["refinements"][-1]
            hestia["soft_policy"]["sensitivity_evidence"].update(
                file_record(
                    Path(raw),
                    "refinement/hestia-sensitivity.json",
                    sensitivity_path.read_bytes(),
                )
            )
            hestia["refinement_id"] = digest(
                {key: value for key, value in hestia.items() if key != "refinement_id"}
            )
            write_json(state["trace_path"], state["trace"])

            with self.assertRaisesRegex(Stage7Error, "source rank-2 tensor inventory"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hestia_sensitivity_not_bound_to_calibration_token_stream(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            sensitivity_path = Path(raw) / "refinement/hestia-sensitivity.json"
            sensitivity = json.loads(sensitivity_path.read_bytes())
            sensitivity["s2kf_token_stream_digest"] = "sha256:" + "0" * 64
            sensitivity["evidence_id"] = digest(
                {
                    key: value
                    for key, value in sensitivity.items()
                    if key != "evidence_id"
                }
            )
            write_json(sensitivity_path, sensitivity)
            hestia = state["trace"]["refinements"][-1]
            hestia["soft_policy"]["sensitivity_evidence"].update(
                file_record(
                    Path(raw),
                    "refinement/hestia-sensitivity.json",
                    sensitivity_path.read_bytes(),
                )
            )
            hestia["refinement_id"] = digest(
                {key: value for key, value in hestia.items() if key != "refinement_id"}
            )
            write_json(state["trace_path"], state["trace"])

            with self.assertRaisesRegex(Stage7Error, "S2KF provenance differs"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hash_consistent_non_salt_refinement_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            refinement = state["trace"]["refinements"][1]
            checkpoint = refinement["checkpoints"][0]
            artifact_path = Path(raw) / checkpoint["artifact"]["path"]
            artifact_path.write_bytes(b"x" * artifact_path.stat().st_size)
            checkpoint["artifact"].update(
                file_record(
                    Path(raw), checkpoint["artifact"]["path"], artifact_path.read_bytes()
                )
            )
            refinement["refinement_id"] = digest(
                {key: value for key, value in refinement.items() if key != "refinement_id"}
            )
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "semantic verification failed"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_refinement_metrics_not_bound_to_checkpoint_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            refinement = state["trace"]["refinements"][1]
            checkpoint = refinement["checkpoints"][0]
            evaluation_path = Path(raw) / checkpoint["evaluation_receipt"]["path"]
            evaluation = json.loads(evaluation_path.read_bytes())
            evaluation["artifact_sha256"] = "0" * 64
            evaluation["evaluation_id"] = digest(
                {
                    key: value
                    for key, value in evaluation.items()
                    if key != "evaluation_id"
                }
            )
            write_json(evaluation_path, evaluation)
            checkpoint["evaluation_receipt"].update(
                file_record(
                    Path(raw),
                    checkpoint["evaluation_receipt"]["path"],
                    evaluation_path.read_bytes(),
                )
            )
            refinement["refinement_id"] = digest(
                {key: value for key, value in refinement.items() if key != "refinement_id"}
            )
            write_json(state["trace_path"], state["trace"])

            with self.assertRaisesRegex(Stage7Error, "checkpoint artifact and metrics"):
                qualify(
                    state["campaign_path"],
                    state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_smoke_execution_stage_assertion_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            execution_path = Path(raw) / "smoke/execution.json"
            execution = json.loads(execution_path.read_bytes())
            execution["stages"][-1]["result"] = "skip"
            write_json(execution_path, execution)
            smoke_path = Path(raw) / "smoke-receipt.json"
            smoke = json.loads(smoke_path.read_bytes())
            smoke["execution_log"].update(
                file_record(Path(raw), "smoke/execution.json", execution_path.read_bytes())
            )
            write_json(smoke_path, smoke)
            state["campaign"]["evidence"][0].update(
                file_record(Path(raw), smoke_path.name, smoke_path.read_bytes())
            )
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "stage order or result"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_hash_consistent_non_salt_smoke_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            smoke_path = Path(raw) / "smoke-receipt.json"
            smoke = json.loads(smoke_path.read_bytes())
            artifact_path = Path(raw) / smoke["artifact"]["path"]
            artifact_path.write_bytes(b"x" * artifact_path.stat().st_size)
            smoke["artifact"].update(
                file_record(
                    Path(raw), smoke["artifact"]["path"], artifact_path.read_bytes()
                )
            )
            execution_path = Path(raw) / smoke["execution_log"]["path"]
            execution = json.loads(execution_path.read_bytes())
            execution["artifact_sha256"] = smoke["artifact"]["sha256"]
            write_json(execution_path, execution)
            smoke["execution_log"].update(
                file_record(
                    Path(raw), smoke["execution_log"]["path"], execution_path.read_bytes()
                )
            )
            write_json(smoke_path, smoke)
            state["campaign"]["evidence"][0].update(
                file_record(Path(raw), smoke_path.name, smoke_path.read_bytes())
            )
            write_json(state["campaign_path"], state["campaign"])
            state["trace"]["campaign_sha256"] = hashlib.sha256(
                state["campaign_path"].read_bytes()
            ).hexdigest()
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "semantic verification failed"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_rejects_unmatched_salt_v1_baseline_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["trace"]["baselines"]["salt_v1"][1]["physical_bytes"] -= 1
            write_json(state["trace_path"], state["trace"])
            with self.assertRaisesRegex(Stage7Error, "physical report totals"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )

    def test_records_terminal_negative_for_valid_physical_accounting_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            baseline = state["trace"]["baselines"]["salt_v1"][1]
            report_path = Path(raw) / baseline["physical_report"]["path"]
            report = json.loads(report_path.read_bytes())
            report["result"] = "negative"
            report["dense_materialized_bytes"] = 1
            write_json(report_path, report)
            baseline["physical_report"].update(
                file_record(
                    Path(raw),
                    baseline["physical_report"]["path"],
                    report_path.read_bytes(),
                )
            )
            write_json(state["trace_path"], state["trace"])

            result = qualify(
                state["campaign_path"],
                state["trace_path"],
                model_root=state["model_root"],
                smoke_model_root=state["smoke_model_root"],
                source_root=state["source"],
            )

            self.assertEqual(result["result"], "negative")
            self.assertIn("physical-accounting-gate-failed", result["freeze_reasons"])

    def test_records_terminal_negative_for_nonselected_full_model_physical_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            full = state["trace"]["stages"][2]
            row = next(
                row
                for row in full["measurements"]
                if state["grid"][row["candidate_id"]]["curvature"] == "input-hessian"
            )
            row["correct"] = False
            report_path = Path(raw) / row["physical_report"]["path"]
            report = json.loads(report_path.read_bytes())
            report["result"] = "negative"
            report["dense_materialized_bytes"] = 1
            write_json(report_path, report)
            row["physical_report"].update(
                file_record(
                    Path(raw), row["physical_report"]["path"], report_path.read_bytes()
                )
            )
            write_json(state["trace_path"], state["trace"])

            result = qualify(
                state["campaign_path"],
                state["trace_path"],
                model_root=state["model_root"],
                smoke_model_root=state["smoke_model_root"],
                source_root=state["source"],
            )

            self.assertEqual(result["result"], "negative")
            self.assertIn("physical-accounting-gate-failed", result["freeze_reasons"])

    def test_records_negative_without_freeze_when_matched_curvature_does_not_win(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            full = state["trace"]["stages"][2]["measurements"]
            by_id = {row["candidate_id"]: row for row in full}
            for identity, target in state["codec_targets"].items():
                control = state["codec_controls"][identity]
                by_id[target]["heldout_ppl"] = by_id[control]["heldout_ppl"]
            write_json(state["trace_path"], state["trace"])
            result = qualify(
                state["campaign_path"], state["trace_path"],
                model_root=state["model_root"],
                smoke_model_root=state["smoke_model_root"],
                source_root=state["source"],
            )
            self.assertEqual(result["result"], "negative")
            self.assertEqual(result["frozen_ptq_recipe_ids"], {})
            self.assertIn(
                "no-configuration-matched-output-aware-curvature-win",
                result["freeze_reasons"],
            )

    def test_records_terminal_negative_when_one_rate_has_no_correct_full_model_point(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            full = state["trace"]["stages"][2]
            r2_ids = {
                candidate_id
                for candidate_id in full["input_ids"]
                if state["grid"][candidate_id]["profile"] == RATE_PROFILE["R2"]
            }
            for row in full["measurements"]:
                if row["candidate_id"] in r2_ids:
                    row["correct"] = False
            full["promoted_ids"] = [
                candidate_id
                for candidate_id in full["promoted_ids"]
                if candidate_id not in r2_ids
            ]
            state["trace"]["refinements"] = []
            write_json(state["trace_path"], state["trace"])

            result = qualify(
                state["campaign_path"],
                state["trace_path"],
                model_root=state["model_root"],
                smoke_model_root=state["smoke_model_root"],
                source_root=state["source"],
            )

            self.assertEqual(result["result"], "negative")
            self.assertFalse(result["freeze_authorized"])
            self.assertEqual(result["freeze_reasons"], ["missing-nondominated-ptq-rate"])
            self.assertEqual(result["frozen_ptq_recipe_ids"], {})
            self.assertIsNone(result["frozen_refined_recipe_id"])
            self.assertEqual(
                result["soft_method_ab"],
                {"outcome": "not-run", "winner": None, "reason": "missing-rate"},
            )

    def test_rejects_dot_model_path_without_crashing(self):
        with tempfile.TemporaryDirectory() as raw:
            state = fixture(Path(raw))
            state["campaign"]["model"]["files"][0]["path"] = "."
            write_json(state["campaign_path"], state["campaign"])
            with self.assertRaisesRegex(Stage7Error, "nonempty POSIX path"):
                qualify(
                    state["campaign_path"], state["trace_path"],
                    model_root=state["model_root"],
                    smoke_model_root=state["smoke_model_root"],
                    source_root=state["source"],
                )


if __name__ == "__main__":
    unittest.main()
