#!/usr/bin/env python3
"""Generate Tritium's sealed Qwen3.5 MTP fixture with pinned vLLM CUDA.

This is the canonical packaged generator; `tools/gen_qwen35_mtp_goldens.py`
is only a workspace launcher.

The generator creates the exact synthetic HF source consumed by the Rust
integration test, verifies its content-derived identity through Tritium, then
executes the real pinned vLLM target and MTP wrappers. Decoder, attention,
cache, and logits execution are unstubbed; shifted IDs and positions come from
the pinned EagleProposer first-pass method and are recorded explicitly.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import ctypes
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import struct
import sys
import sysconfig
import tempfile
import urllib.parse
from pathlib import Path

_FORBIDDEN_INHERITED_RUNTIME_VARIABLES = frozenset(
    {
        "CC",
        "DISABLE_PTXAS_OPT",
        "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
        "NVPTX_ENABLE_DUMP",
        "PTXAS_OPTIONS",
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
        "USE_IR_LOC",
    }
)
_forbidden_inherited_runtime_variables = sorted(
    name
    for name, value in os.environ.items()
    if value
    and (
        name.startswith("TRITON_")
        or name in _FORBIDDEN_INHERITED_RUNTIME_VARIABLES
    )
)
if _forbidden_inherited_runtime_variables:
    raise SystemExit(
        "inherited compiler/runtime overrides are forbidden: "
        + ", ".join(_forbidden_inherited_runtime_variables)
    )

# Never consume a developer's pre-existing Triton/vLLM cache as oracle input.
# The generated JIT artifacts are hashed into the implementation manifest below.
_RUNTIME_CACHE_ROOT = Path(tempfile.mkdtemp(prefix="tritium-mtp-oracle-cache-"))
os.environ["TRITON_CACHE_DIR"] = str(_RUNTIME_CACHE_ROOT / "triton")
os.environ["VLLM_CACHE_ROOT"] = str(_RUNTIME_CACHE_ROOT / "vllm")
os.environ["XDG_CACHE_HOME"] = str(_RUNTIME_CACHE_ROOT / "xdg")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_RUNTIME_CACHE_ROOT / "inductor")
os.environ["CC"] = "/usr/bin/gcc"
atexit.register(shutil.rmtree, _RUNTIME_CACHE_ROOT, ignore_errors=True)

import blake3
import torch
import triton
from safetensors.torch import save_file
from triton import knobs as triton_knobs

import vllm
import vllm.model_executor.models.qwen3_5 as qwen35_source
import vllm.model_executor.models.qwen3_5_mtp as qwen35_mtp_source
import vllm.model_executor.models.qwen3_next as qwen3_next_source
import vllm.model_executor.layers.logits_processor as logits_processor_source
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_parallel_source
import vllm.v1.attention.backends.triton_attn as triton_attn_source
import vllm.v1.spec_decode.eagle as eagle_source
import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer_source
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.utils import bind_kv_cache


REVISION = "36484e464a6cf763c5b4c8af7be8e19df324997a"
VLLM_TREE = "b7060f697d63047bcb4e853a968082c91bb46339"
TRANSFORMERS_VERSION = "5.13.1"
TORCH_VERSION = "2.11.0+cu130"
TORCH_GIT_REVISION = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
TRITON_VERSION = "3.6.0"
CUDA_VERSION = "13.0"
CUDA_DRIVER_API_VERSION = 13030
GPU_COMPUTE_CAPABILITY = (8, 9)
GPU_NAME = "NVIDIA GeForce RTX 4090"
GPU_SM_COUNT = 128
GPU_MEMORY_BYTES = 25_271_402_496
NVIDIA_DRIVER_VERSION = "610.43.03"
NVML_VERSION = "13.610.43.03"
VLLM_DISTRIBUTION_VERSION = "0.1.dev1+g36484e464.precompiled"
ATTENTION_BACKEND = "TRITON_ATTN"
EXPECTED_DRIVER_LIBRARY_SHA256 = {
    "libcuda.so.1": "ba35b4baccf427f74f1b7c600297ae0cd4ff860f381aba65c3d0b88b8c5e95bc",
    "libnvidia-ml.so.1": "ad88c4a3e6c72b30d2f59f06d404abab119eb8a24a6785e3a55b4abcda8bfc6b",
}
EXPECTED_COMPILER_SHA256 = {
    "gcc": "12967a6f8b9d20e659d09a24a1d06e79d773c6aa836a7aec4900974dcca1502d",
    "triton_libdevice": "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee",
    "triton_ptxas": "c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca",
}
EXPECTED_TRITON_COMPILER_CONTROLS = {
    "ptxas_selection": "bundled",
    "ptxas_version": "12.8",
    "libdevice_override": None,
    "libcuda_override": None,
    "cudacrt_override": None,
    "cudart_override": None,
    "override_arch": None,
    "fp32_default": None,
    "default_fp_fusion": True,
    "disable_ptxas_opt": False,
    "ptxas_options": None,
    "mock_ptx_version": None,
}
EXPECTED_LOADED_VLLM_NATIVE_SHA256 = {
    "_C_stable_libtorch.abi3.so": "8e41d1c0e9ad5429c4cdeae8567a3e0ae2baeedbf8df8bbdc6913a4c3db6693a",
    "_moe_C_stable_libtorch.abi3.so": "3c553a7e87de7b2b6d79ef977d9f30176a804563301a9169f8a9b22499688d33",
    "_qutlass_C.abi3.so": "43185ce6a559eab1fe8d75c8674a7260ea0a9eed1282274e169d3dbecdc36ea5",
    "vllm_flash_attn/_vllm_fa2_C.abi3.so": "64dff504b9ea5e88c08b3d2efd82f5d7a53dc48cc8be0a4744c92960bcf35478",
    "vllm_flash_attn/_vllm_fa3_C.abi3.so": "af90d0e144bddb52b4b790fe1e01c08d019d53021f5efbfcc245ec8b439258f2",
}
EXPECTED_TRITON_CUBIN_SHA256 = [
    "1a39ae84ba0a968bd314323333e3a8cd49757025478f76135b8434dbc8cce523",
    "386ca3cae4ba88c9f69e3db1c96da27656e952099d20aecf00091cdfbf43733f",
    "503dd7d1394e181e7d9e8a41a185b2b5f60f07ebe148b375505dc801246cca14",
    "526877e8bdab0348b1c2f8537e88b425bc88a8a31713b446ed05c762c2e50a59",
    "6cb80c13d29ebcb2237253ef422e3f09212cfe8755ccecf354f69579033341ef",
    "6faeebe91591f3e1c8a40bf6e242234f6ef3c4ccead4c0fe7bae63a256c59a9f",
    "f862a3875e36b635eae021960b8f2a9770443b12dd07b193ed515b26daa600eb",
    "fbb6e63e2d99b8213081a5b13fd012bbcbd2bda7275a89c25c65aa3e8e59df52",
]
SOURCE_HASHES = {
    "qwen3_5_mtp.py": "87cff3c5ca1c9c6dde87e69b298697d342fb50f1a926e16b59d9e9a9fadb3cc8",
    "qwen3_5.py": "c1d75768f29b054802c629e890b899b6fd8508eaa682383a72af433cf1d08a38",
    "qwen3_next.py": "97b502c29a6cef62716f3e65ed8508017a67fb6d33aa1c4080cbd3e3c2f75099",
    "logits_processor.py": "d6dfec7287020d587f4a4d475ce113c383b2eb55afe5ae9b0e037df2e2d13310",
    "vocab_parallel_embedding.py": (
        "60b116e260d5438513c9bb60370306eed5e9b0ce612bfbe0a3885e64fdf0587b"
    ),
    "triton_attn.py": "6fe27702d44f9cef76c4263042ab3655b1f669224b02abea858437513a7ab7cd",
    "llm_base_proposer.py": "873e18f36749cc4bf35e6cab4755763bfd773d773970b5a9a0bff81e66e65275",
    "eagle.py": "b2b1f7d15117b43108be368756d9c6e9334d9415a070e46178e18bb7c6476378",
}

HIDDEN = 32
INTERMEDIATE = 48
VOCAB = 37
QUERY_HEADS = 2
KV_HEADS = 2
HEAD_DIM = 32
BLOCK_SIZE = 16
ARTIFACT_MAX_CONTEXT = 16
FIXTURE_SOURCE_MODEL_ID = (
    "e79eeacd416d7ce9d4f223955b189bcc90ebb41882619f4c4b3b880465126d92"
)
FIXTURE_SOURCE_CONFIG_DIGEST = (
    "c9b63f23be99ae8ff23b827fba6be653efbaff68a2eeb8dc0dbed753b8b8f39c"
)
MANIFEST_ID_CONTEXT = "tritium qwen3.5 mtp oracle implementation manifest v1"
BODY_ID_CONTEXT = "tritium qwen3.5 mtp oracle body v1"


def parameter(ordinal: int, length: int) -> torch.Tensor:
    return torch.tensor(
        [
            (((17 * index + 13 * ordinal + 5) % 29) - 14) / 32
            for index in range(length)
        ],
        dtype=torch.float32,
    )


def matrix(ordinal: int, rows: int, columns: int) -> torch.Tensor:
    return parameter(ordinal, rows * columns).reshape(rows, columns)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_files(root: Path, predicate=lambda _path: True) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and predicate(path)
    }


def distribution_identity(name: str) -> dict[str, object]:
    distribution = importlib.metadata.distribution(name)
    metadata_names = {"INSTALLER", "METADATA", "WHEEL"}
    if name != "vllm":
        metadata_names.add("RECORD")
    metadata_hashes: dict[str, str] = {}
    for entry in distribution.files or ():
        rendered = str(entry)
        if ".dist-info/" not in rendered or Path(rendered).name not in metadata_names:
            continue
        path = Path(distribution.locate_file(entry))
        metadata_hashes[rendered] = sha256(path)
    if not metadata_hashes:
        raise SystemExit(f"distribution {name} exposes no wheel metadata")
    identity: dict[str, object] = {
        "version": distribution.version,
        "metadata_sha256": metadata_hashes,
    }
    if name != "vllm":
        record_entries = [
            entry
            for entry in distribution.files or ()
            if Path(str(entry)).name == "RECORD"
            and ".dist-info/" in str(entry)
        ]
        if len(record_entries) != 1:
            raise SystemExit(f"distribution {name} must expose one RECORD")
        verified_entries = 0
        verified_bytes = 0
        with Path(distribution.locate_file(record_entries[0])).open(
            newline=""
        ) as record:
            for relative, encoded_digest, encoded_size in csv.reader(record):
                if not encoded_digest:
                    continue
                algorithm, expected_digest = encoded_digest.split("=", 1)
                path = Path(distribution.locate_file(relative))
                expected_size = int(encoded_size)
                observed_size = path.stat().st_size
                if observed_size != expected_size:
                    raise SystemExit(
                        f"distribution {name} payload size mismatch for {relative}: "
                        f"expected {expected_size}, found {observed_size}"
                    )
                digest = hashlib.new(algorithm)
                with path.open("rb") as payload:
                    while chunk := payload.read(1024 * 1024):
                        digest.update(chunk)
                observed_digest = base64.urlsafe_b64encode(digest.digest()).decode()
                observed_digest = observed_digest.rstrip("=")
                if observed_digest != expected_digest:
                    raise SystemExit(
                        f"distribution {name} payload digest mismatch for {relative}"
                    )
                verified_entries += 1
                verified_bytes += observed_size
        identity["record_payload_verification"] = {
            "hashed_entries": verified_entries,
            "hashed_bytes": verified_bytes,
        }
    return identity


def mapped_library_path(name: str) -> Path:
    matches = set()
    shared_object_stem = name.split(".so", 1)[0] + ".so"
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5].endswith(" (deleted)"):
            continue
        path = Path(fields[5])
        if path.name == shared_object_stem or path.name.startswith(
            shared_object_stem + "."
        ):
            matches.add(path.resolve())
    if len(matches) != 1:
        raise SystemExit(f"expected one mapped {name}, found {sorted(matches)}")
    return matches.pop()


def driver_identity() -> dict[str, object]:
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise SystemExit("cuInit failed while sealing CUDA driver identity")
    driver_api = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(driver_api)) != 0:
        raise SystemExit("cuDriverGetVersion failed")

    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    if nvml.nvmlInit_v2() != 0:
        raise SystemExit("nvmlInit_v2 failed")
    try:
        driver = ctypes.create_string_buffer(80)
        nvml_version = ctypes.create_string_buffer(80)
        if nvml.nvmlSystemGetDriverVersion(driver, len(driver)) != 0:
            raise SystemExit("nvmlSystemGetDriverVersion failed")
        if nvml.nvmlSystemGetNVMLVersion(nvml_version, len(nvml_version)) != 0:
            raise SystemExit("nvmlSystemGetNVMLVersion failed")
    finally:
        nvml.nvmlShutdown()

    observed = {
        "cuda_driver_api_version": driver_api.value,
        "nvidia_driver_version": driver.value.decode(),
        "nvml_version": nvml_version.value.decode(),
        "driver_library_sha256": {
            name: sha256(mapped_library_path(name))
            for name in EXPECTED_DRIVER_LIBRARY_SHA256
        },
    }
    expected = {
        "cuda_driver_api_version": CUDA_DRIVER_API_VERSION,
        "nvidia_driver_version": NVIDIA_DRIVER_VERSION,
        "nvml_version": NVML_VERSION,
        "driver_library_sha256": EXPECTED_DRIVER_LIBRARY_SHA256,
    }
    if observed != expected:
        raise SystemExit(f"CUDA driver identity mismatch: {observed}")
    return observed


def compiler_identity() -> dict[str, str]:
    triton_root = Path(triton.__file__).resolve().parent
    gcc = Path(os.environ["CC"]).resolve()
    if not gcc.is_file():
        raise SystemExit(f"configured launcher compiler does not exist: {gcc}")
    paths = {
        "gcc": gcc,
        "triton_libdevice": triton_root
        / "backends"
        / "nvidia"
        / "lib"
        / "libdevice.10.bc",
        "triton_ptxas": triton_root / "backends" / "nvidia" / "bin" / "ptxas",
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    if observed != EXPECTED_COMPILER_SHA256:
        raise SystemExit(f"Triton compiler identity mismatch: {observed}")
    return observed


def triton_compiler_controls() -> dict[str, str | bool | None]:
    bundled_ptxas = (
        Path(triton.__file__).resolve().parent
        / "backends"
        / "nvidia"
        / "bin"
        / "ptxas"
    ).resolve()
    selected_ptxas = Path(triton_knobs.nvidia.ptxas.path).resolve()
    observed = {
        "ptxas_selection": (
            "bundled" if selected_ptxas == bundled_ptxas else str(selected_ptxas)
        ),
        "ptxas_version": triton_knobs.nvidia.ptxas.version,
        "libdevice_override": triton_knobs.nvidia.libdevice_path,
        "libcuda_override": triton_knobs.nvidia.libcuda_path,
        "cudacrt_override": triton_knobs.build.cudacrt_path,
        "cudart_override": triton_knobs.build.cudart_path,
        "override_arch": triton_knobs.runtime.override_arch,
        "fp32_default": triton_knobs.language.fp32_default,
        "default_fp_fusion": triton_knobs.language.default_fp_fusion,
        "disable_ptxas_opt": triton_knobs.nvidia.disable_ptxas_opt,
        "ptxas_options": triton_knobs.nvidia.ptxas_options,
        "mock_ptx_version": triton_knobs.nvidia.mock_ptx_version,
    }
    if observed != EXPECTED_TRITON_COMPILER_CONTROLS:
        raise SystemExit(f"Triton compiler controls mismatch: {observed}")
    return observed


def loaded_vllm_native_extensions(repo: Path) -> dict[str, str]:
    package = (repo / "vllm").resolve()
    observed: dict[str, str] = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5].endswith(" (deleted)"):
            continue
        path = Path(fields[5]).resolve()
        try:
            relative = path.relative_to(package).as_posix()
        except ValueError:
            continue
        if ".so" in path.name:
            observed[relative] = sha256(path)
    if observed != EXPECTED_LOADED_VLLM_NATIVE_SHA256:
        raise SystemExit(f"loaded vLLM native extension mismatch: {observed}")
    return observed


def loaded_numeric_libraries() -> dict[str, str]:
    environment = Path(sys.prefix).resolve()
    package_prefixes = (
        "blake3",
        "numpy",
        "nvidia",
        "safetensors",
        "torch",
        "triton",
    )
    system_prefixes = (
        "libcuda.so",
        "libcublas",
        "libcudnn",
        "libcufft",
        "libcurand",
        "libcusolver",
        "libcusparse",
        "libnccl",
        "libnvidia",
        "libnvrtc",
        "libnvshmem",
    )
    observed: dict[str, str] = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5].endswith(" (deleted)"):
            continue
        path = Path(fields[5]).resolve()
        if ".so" not in path.name or not path.is_file():
            continue
        key = None
        try:
            relative = path.relative_to(environment)
        except ValueError:
            relative = None
        if relative is not None and "site-packages" in relative.parts:
            package_index = relative.parts.index("site-packages") + 1
            if package_index < len(relative.parts) and relative.parts[
                package_index
            ].startswith(package_prefixes):
                key = f"python-environment/{relative.as_posix()}"
        if key is None and path.name.startswith(system_prefixes):
            key = f"system/{path.name}"
        if key is None:
            continue
        digest = sha256(path)
        if key in observed and observed[key] != digest:
            raise SystemExit(f"ambiguous mapped numeric library identity for {key}")
        observed[key] = digest
    if not observed:
        raise SystemExit("no mapped numeric libraries were sealed")
    return observed


def triton_cubin_hashes() -> list[str]:
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    cubins = sorted(sha256(path) for path in cache.rglob("*.cubin"))
    if cubins != EXPECTED_TRITON_CUBIN_SHA256:
        raise SystemExit(f"generated Triton cubin identity mismatch: {cubins}")
    return cubins


def fixture_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": False,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "full_attention_interval": 1,
            "head_dim": HEAD_DIM,
            "hidden_act": "silu",
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "layer_types": ["full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 8,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 64,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": QUERY_HEADS,
            "num_hidden_layers": 1,
            "num_key_value_heads": KV_HEADS,
            "output_gate_type": "swish",
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [8, 0, 0],
                "partial_rotary_factor": 0.5,
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": VOCAB,
        },
        "tie_word_embeddings": False,
        "vision_config": {"model_type": "qwen3_5"},
    }


def fixture_tensors() -> list[tuple[str, torch.Tensor]]:
    query_width = QUERY_HEADS * HEAD_DIM
    gated_query_width = 2 * query_width
    kv_width = KV_HEADS * HEAD_DIM
    language = "model.language_model.layers.0"
    mtp = "mtp.layers.0"
    return [
        ("model.language_model.embed_tokens.weight", matrix(0, VOCAB, HIDDEN)),
        (
            f"{language}.self_attn.q_proj.weight",
            matrix(1, gated_query_width, HIDDEN),
        ),
        (
            f"{language}.self_attn.k_proj.weight",
            matrix(2, kv_width, HIDDEN),
        ),
        (
            f"{language}.self_attn.v_proj.weight",
            matrix(3, kv_width, HIDDEN),
        ),
        (
            f"{language}.self_attn.o_proj.weight",
            matrix(4, HIDDEN, query_width),
        ),
        (f"{language}.self_attn.q_norm.weight", parameter(5, HEAD_DIM)),
        (f"{language}.self_attn.k_norm.weight", parameter(6, HEAD_DIM)),
        (
            f"{language}.mlp.gate_proj.weight",
            matrix(7, INTERMEDIATE, HIDDEN),
        ),
        (
            f"{language}.mlp.up_proj.weight",
            matrix(8, INTERMEDIATE, HIDDEN),
        ),
        (
            f"{language}.mlp.down_proj.weight",
            matrix(9, HIDDEN, INTERMEDIATE),
        ),
        (f"{language}.input_layernorm.weight", parameter(10, HIDDEN)),
        (
            f"{language}.post_attention_layernorm.weight",
            parameter(11, HIDDEN),
        ),
        ("model.language_model.norm.weight", parameter(12, HIDDEN)),
        ("lm_head.weight", matrix(13, VOCAB, HIDDEN)),
        ("model.visual.pos_embed.weight", matrix(14, 1, HIDDEN)),
        ("mtp.pre_fc_norm_embedding.weight", parameter(15, HIDDEN)),
        ("mtp.pre_fc_norm_hidden.weight", parameter(16, HIDDEN)),
        ("mtp.fc.weight", matrix(17, HIDDEN, 2 * HIDDEN)),
        (f"{mtp}.input_layernorm.weight", parameter(18, HIDDEN)),
        (
            f"{mtp}.self_attn.q_proj.weight",
            matrix(19, gated_query_width, HIDDEN),
        ),
        (f"{mtp}.self_attn.k_proj.weight", matrix(20, kv_width, HIDDEN)),
        (f"{mtp}.self_attn.v_proj.weight", matrix(21, kv_width, HIDDEN)),
        (f"{mtp}.self_attn.o_proj.weight", matrix(22, HIDDEN, query_width)),
        (f"{mtp}.self_attn.q_norm.weight", parameter(23, HEAD_DIM)),
        (f"{mtp}.self_attn.k_norm.weight", parameter(24, HEAD_DIM)),
        (f"{mtp}.post_attention_layernorm.weight", parameter(25, HIDDEN)),
        (f"{mtp}.mlp.gate_proj.weight", matrix(26, INTERMEDIATE, HIDDEN)),
        (f"{mtp}.mlp.up_proj.weight", matrix(27, INTERMEDIATE, HIDDEN)),
        (f"{mtp}.mlp.down_proj.weight", matrix(28, HIDDEN, INTERMEDIATE)),
        ("mtp.norm.weight", parameter(29, HIDDEN)),
    ]


def write_fixture_source(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(
        json.dumps(fixture_config(), sort_keys=True, separators=(",", ":")) + "\n"
    )
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    shards: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] = ({}, {})
    weight_map: dict[str, str] = {}
    for index, (name, value) in enumerate(fixture_tensors()):
        shard_index = index % 2
        shards[shard_index][name] = value.contiguous()
        weight_map[name] = shard_names[shard_index]
    for name, tensors in zip(shard_names, shards, strict=True):
        save_file(tensors, directory / name)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def verify_fixture_identity(directory: Path, identity_bin: Path) -> dict[str, str]:
    identity_bin = identity_bin.resolve()
    if not identity_bin.is_file():
        raise SystemExit(
            f"missing Rust identity helper {identity_bin}; run "
            "`cargo build -p tritium-nn --example qwen35_hf_identity`"
        )
    raw = subprocess.check_output([str(identity_bin), str(directory)], text=True)
    identity = json.loads(raw)
    expected = {
        "source_model_id": FIXTURE_SOURCE_MODEL_ID,
        "source_config_digest": FIXTURE_SOURCE_CONFIG_DIGEST,
    }
    if identity != expected:
        raise SystemExit(
            f"fixture identity mismatch: expected {expected}, observed {identity}"
        )
    return identity


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def verify_sources(
    required_repo: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    modules = {
        "qwen3_5_mtp.py": qwen35_mtp_source,
        "qwen3_5.py": qwen35_source,
        "qwen3_next.py": qwen3_next_source,
        "logits_processor.py": logits_processor_source,
        "vocab_parallel_embedding.py": vocab_parallel_source,
        "triton_attn.py": triton_attn_source,
        "llm_base_proposer.py": llm_base_proposer_source,
        "eagle.py": eagle_source,
    }
    repo = Path(vllm.__file__).resolve().parent.parent
    required_repo = required_repo.resolve()
    if repo != required_repo:
        raise SystemExit(
            f"vLLM import root mismatch: required {required_repo}, imported {repo}"
        )
    observed_revision = git_output(repo, "rev-parse", "HEAD")
    if observed_revision != REVISION:
        raise SystemExit(
            f"vLLM checkout mismatch: expected {REVISION}, observed {observed_revision}"
        )
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(f"vLLM checkout has tracked modifications:\n{dirty}")

    paths: dict[str, str] = {}
    for name, module in modules.items():
        path = Path(inspect.getsourcefile(module) or "")
        observed = sha256(path)
        expected = SOURCE_HASHES[name]
        if observed != expected:
            raise SystemExit(
                f"{name} source mismatch: expected {expected}, observed {observed}"
            )
        paths[name] = str(path)
        try:
            path.relative_to(repo)
        except ValueError as error:
            raise SystemExit(f"{name} imported outside pinned checkout: {path}") from error
    observed_tree = git_output(repo, "rev-parse", "HEAD^{tree}")
    if observed_tree != VLLM_TREE:
        raise SystemExit(
            f"vLLM tree mismatch: expected {VLLM_TREE}, observed {observed_tree}"
        )
    if "g36484e464" not in vllm.__version__:
        raise SystemExit(f"unexpected vLLM version {vllm.__version__}")
    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != VLLM_DISTRIBUTION_VERSION:
        raise SystemExit(
            "unexpected vLLM distribution version "
            f"{distribution.version}, expected {VLLM_DISTRIBUTION_VERSION}"
        )
    direct_url_entries = [
        entry
        for entry in distribution.files or ()
        if Path(str(entry)).name == "direct_url.json"
    ]
    if len(direct_url_entries) != 1:
        raise SystemExit("vLLM distribution must expose one direct_url.json")
    direct_url = json.loads(
        Path(distribution.locate_file(direct_url_entries[0])).read_text()
    )
    parsed_url = urllib.parse.urlparse(direct_url.get("url", ""))
    installed_source = Path(urllib.parse.unquote(parsed_url.path)).resolve()
    if (
        parsed_url.scheme != "file"
        or installed_source != repo
        or direct_url.get("dir_info", {}).get("editable") is not True
    ):
        raise SystemExit(
            "vLLM editable distribution is not bound to the required checkout"
        )
    native_extensions = hashed_files(
        repo / "vllm", lambda path: ".so" in path.name
    )
    if not native_extensions:
        raise SystemExit("pinned vLLM checkout exposes no native extensions")
    return paths, {
        "repo": str(repo),
        "commit": observed_revision,
        "tree": observed_tree,
        "tracked_worktree_clean": True,
        "native_extensions_sha256": native_extensions,
    }


def tiny_text_config(_: object) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=QUERY_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        full_attention_interval=1,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
        mamba_ssm_dtype="float32",
        output_gate_type="swish",
        partial_rotary_factor=0.5,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [8, 0, 0],
        },
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3_5MTP"]
    config.mtp_num_hidden_layers = 1
    config.mtp_use_dedicated_embeddings = False
    return config


def make_vllm_config(snapshot: Path, backend: str) -> VllmConfig:
    parallel = ParallelConfig()
    model = ModelConfig(
        model=str(snapshot),
        tokenizer=str(snapshot),
        runner="generate",
        max_model_len=32,
        dtype="float32",
        hf_overrides=tiny_text_config,
    )
    return VllmConfig(
        model_config=model,
        parallel_config=parallel,
        device_config=DeviceConfig("cuda"),
        attention_config=AttentionConfig(backend=backend),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=False,
        ),
    )


def make_proposer_config(snapshot: Path, backend: str) -> VllmConfig:
    parallel = ParallelConfig()
    target = ModelConfig(
        model=str(snapshot),
        tokenizer=str(snapshot),
        runner="generate",
        max_model_len=32,
        dtype="float32",
    )
    speculative = SpeculativeConfig(
        num_speculative_tokens=1,
        method="mtp",
        target_model_config=target,
        target_parallel_config=parallel,
    )
    return VllmConfig(
        model_config=target,
        speculative_config=speculative,
        parallel_config=parallel,
        device_config=DeviceConfig("cuda"),
        attention_config=AttentionConfig(backend=backend),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=False,
        ),
    )
def target_weights() -> list[tuple[str, torch.Tensor]]:
    # One full-attention target layer plus final language RMSNorm.
    return [
        ("embed_tokens.weight", matrix(0, VOCAB, HIDDEN)),
        ("layers.0.input_layernorm.weight", parameter(10, HIDDEN)),
        (
            "layers.0.self_attn.q_proj.weight",
            matrix(1, 2 * QUERY_HEADS * HEAD_DIM, HIDDEN),
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            matrix(2, KV_HEADS * HEAD_DIM, HIDDEN),
        ),
        (
            "layers.0.self_attn.v_proj.weight",
            matrix(3, KV_HEADS * HEAD_DIM, HIDDEN),
        ),
        (
            "layers.0.self_attn.o_proj.weight",
            matrix(4, HIDDEN, QUERY_HEADS * HEAD_DIM),
        ),
        ("layers.0.self_attn.q_norm.weight", parameter(5, HEAD_DIM)),
        ("layers.0.self_attn.k_norm.weight", parameter(6, HEAD_DIM)),
        ("layers.0.post_attention_layernorm.weight", parameter(11, HIDDEN)),
        ("layers.0.mlp.gate_proj.weight", matrix(7, INTERMEDIATE, HIDDEN)),
        ("layers.0.mlp.up_proj.weight", matrix(8, INTERMEDIATE, HIDDEN)),
        ("layers.0.mlp.down_proj.weight", matrix(9, HIDDEN, INTERMEDIATE)),
        ("norm.weight", parameter(12, HIDDEN)),
    ]


def mtp_weights() -> list[tuple[str, torch.Tensor]]:
    # These are the exact 15 HF MTP tensors. AutoWeightsLoader folds q/k/v
    # into qkv_proj and gate/up into gate_up_proj, yielding 12 vLLM params.
    return [
        ("pre_fc_norm_embedding.weight", parameter(15, HIDDEN)),
        ("pre_fc_norm_hidden.weight", parameter(16, HIDDEN)),
        ("fc.weight", matrix(17, HIDDEN, 2 * HIDDEN)),
        ("layers.0.input_layernorm.weight", parameter(18, HIDDEN)),
        (
            "layers.0.self_attn.q_proj.weight",
            matrix(19, 2 * QUERY_HEADS * HEAD_DIM, HIDDEN),
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            matrix(20, KV_HEADS * HEAD_DIM, HIDDEN),
        ),
        (
            "layers.0.self_attn.v_proj.weight",
            matrix(21, KV_HEADS * HEAD_DIM, HIDDEN),
        ),
        (
            "layers.0.self_attn.o_proj.weight",
            matrix(22, HIDDEN, QUERY_HEADS * HEAD_DIM),
        ),
        ("layers.0.self_attn.q_norm.weight", parameter(23, HEAD_DIM)),
        ("layers.0.self_attn.k_norm.weight", parameter(24, HEAD_DIM)),
        ("layers.0.post_attention_layernorm.weight", parameter(25, HIDDEN)),
        ("layers.0.mlp.gate_proj.weight", matrix(26, INTERMEDIATE, HIDDEN)),
        ("layers.0.mlp.up_proj.weight", matrix(27, INTERMEDIATE, HIDDEN)),
        ("layers.0.mlp.down_proj.weight", matrix(28, HIDDEN, INTERMEDIATE)),
        ("norm.weight", parameter(29, HIDDEN)),
    ]


def common_metadata(
    *, seq_len: int, query_len: int, device: torch.device
) -> CommonAttentionMetadata:
    query_start = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    return CommonAttentionMetadata(
        query_start_loc=query_start,
        query_start_loc_cpu=query_start.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.tensor(
            [seq_len - query_len], dtype=torch.int32
        ),
        num_reqs=1,
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=seq_len,
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32, device=device),
        slot_mapping=torch.arange(
            seq_len - query_len,
            seq_len,
            dtype=torch.int64,
            device=device,
        ),
        causal=True,
    )


class RuntimeAttention:
    def __init__(self, attention, vllm_config: VllmConfig):
        self.attention = attention
        self.name = attention.layer_name
        self.backend = attention.attn_backend
        self.spec = attention.get_kv_cache_spec(vllm_config)
        if self.spec is None:
            raise RuntimeError(f"{self.name} did not produce a KV-cache spec")
        self.builder = self.backend.get_builder_cls()(
            self.spec,
            [self.name],
            vllm_config,
            torch.device("cuda"),
        )
        shape = self.backend.get_kv_cache_shape(
            1,
            self.spec.block_size,
            self.spec.num_kv_heads,
            self.spec.head_size,
            cache_dtype_str="auto",
        )
        logical = torch.zeros(
            shape,
            dtype=vllm_config.model_config.dtype,
            device="cuda",
        )
        order = self.backend.get_kv_cache_stride_order()
        inverse = [order.index(index) for index in range(len(order))]
        # Same logical shape as get_kv_cache_shape, with runtime physical strides.
        self.cache = logical.permute(*order).contiguous().permute(*inverse)

    def metadata(
        self,
        common: CommonAttentionMetadata,
        *,
        drafting: bool,
        draft_index: int = 0,
    ):
        if drafting:
            return self.builder.build_for_drafting(
                common_attn_metadata=common,
                draft_index=draft_index,
            )
        return self.builder.build(0, common)

    def canonical_kv(self, used_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Logical cache layout is [block, kv_head, token, 2 * head_dim].
        # This fixture uses one block. Tritium's canonical contract is
        # [sequence, kv_head, head_dim], so transpose before flattening.
        head_major = self.cache[0, :, :used_tokens]
        packed = head_major.permute(1, 0, 2).contiguous()
        keys = packed[..., : self.spec.head_size].contiguous()
        values = packed[..., self.spec.head_size :].contiguous()
        if used_tokens > 1 and self.spec.num_kv_heads > 1:
            raw_keys = head_major[..., : self.spec.head_size].contiguous()
            if torch.equal(keys.flatten(), raw_keys.flatten()):
                raise AssertionError("fixture does not distinguish KV cache layouts")
        return keys, values


def as_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().cpu().flatten().tolist()


def assert_all_parameters_loaded(
    label: str, model: torch.nn.Module, loaded: set[str]
) -> None:
    expected = {name for name, _ in model.named_parameters()}
    if loaded != expected:
        missing = sorted(expected - loaded)
        unexpected = sorted(loaded - expected)
        raise AssertionError(
            f"{label} load coverage mismatch; missing={missing}, unexpected={unexpected}"
        )


def padded_vocab(weight: torch.Tensor, rows: int) -> torch.Tensor:
    padded = torch.zeros(rows, HIDDEN, dtype=weight.dtype)
    padded[: weight.shape[0]].copy_(weight)
    return padded


def assert_exact_parameters(
    label: str,
    model: torch.nn.Module,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = dict(model.named_parameters())
    if actual.keys() != expected.keys():
        raise AssertionError(f"{label} exact parameter names do not match")
    for name, wanted in expected.items():
        observed = actual[name]
        wanted = wanted.to(device=observed.device, dtype=observed.dtype)
        if not torch.equal(observed, wanted):
            raise AssertionError(f"{label} exact weight mismatch: {name}")


def expected_target_parameters(target: torch.nn.Module) -> dict[str, torch.Tensor]:
    embedding_rows = dict(target.named_parameters())["model.embed_tokens.weight"].shape[0]
    head_rows = dict(target.named_parameters())["lm_head.weight"].shape[0]
    return {
        "model.embed_tokens.weight": padded_vocab(
            matrix(0, VOCAB, HIDDEN), embedding_rows
        ),
        "model.layers.0.input_layernorm.weight": parameter(10, HIDDEN),
        "model.layers.0.self_attn.qkv_proj.weight": torch.cat(
            [
                matrix(1, 2 * QUERY_HEADS * HEAD_DIM, HIDDEN),
                matrix(2, KV_HEADS * HEAD_DIM, HIDDEN),
                matrix(3, KV_HEADS * HEAD_DIM, HIDDEN),
            ]
        ),
        "model.layers.0.self_attn.o_proj.weight": matrix(
            4, HIDDEN, QUERY_HEADS * HEAD_DIM
        ),
        "model.layers.0.self_attn.q_norm.weight": parameter(5, HEAD_DIM),
        "model.layers.0.self_attn.k_norm.weight": parameter(6, HEAD_DIM),
        "model.layers.0.post_attention_layernorm.weight": parameter(11, HIDDEN),
        "model.layers.0.mlp.gate_up_proj.weight": torch.cat(
            [
                matrix(7, INTERMEDIATE, HIDDEN),
                matrix(8, INTERMEDIATE, HIDDEN),
            ]
        ),
        "model.layers.0.mlp.down_proj.weight": matrix(
            9, HIDDEN, INTERMEDIATE
        ),
        "model.norm.weight": parameter(12, HIDDEN),
        "lm_head.weight": padded_vocab(matrix(13, VOCAB, HIDDEN), head_rows),
    }


def expected_mtp_parameters(mtp: torch.nn.Module) -> dict[str, torch.Tensor]:
    embedding_rows = dict(mtp.named_parameters())["model.embed_tokens.weight"].shape[0]
    head_rows = dict(mtp.named_parameters())["lm_head.weight"].shape[0]
    return {
        "model.embed_tokens.weight": padded_vocab(
            matrix(0, VOCAB, HIDDEN), embedding_rows
        ),
        "model.fc.weight": matrix(17, HIDDEN, 2 * HIDDEN),
        "model.layers.0.input_layernorm.weight": parameter(18, HIDDEN),
        "model.layers.0.self_attn.qkv_proj.weight": torch.cat(
            [
                matrix(19, 2 * QUERY_HEADS * HEAD_DIM, HIDDEN),
                matrix(20, KV_HEADS * HEAD_DIM, HIDDEN),
                matrix(21, KV_HEADS * HEAD_DIM, HIDDEN),
            ]
        ),
        "model.layers.0.self_attn.o_proj.weight": matrix(
            22, HIDDEN, QUERY_HEADS * HEAD_DIM
        ),
        "model.layers.0.self_attn.q_norm.weight": parameter(23, HEAD_DIM),
        "model.layers.0.self_attn.k_norm.weight": parameter(24, HEAD_DIM),
        "model.layers.0.post_attention_layernorm.weight": parameter(25, HIDDEN),
        "model.layers.0.mlp.gate_up_proj.weight": torch.cat(
            [
                matrix(26, INTERMEDIATE, HIDDEN),
                matrix(27, INTERMEDIATE, HIDDEN),
            ]
        ),
        "model.layers.0.mlp.down_proj.weight": matrix(
            28, HIDDEN, INTERMEDIATE
        ),
        "model.norm.weight": parameter(29, HIDDEN),
        "model.pre_fc_norm_embedding.weight": parameter(15, HIDDEN),
        "model.pre_fc_norm_hidden.weight": parameter(16, HIDDEN),
        "lm_head.weight": padded_vocab(matrix(13, VOCAB, HIDDEN), head_rows),
    }


def verify_runtime() -> dict[str, object]:
    versions = {
        "transformers": (importlib.metadata.version("transformers"), TRANSFORMERS_VERSION),
        "torch": (torch.__version__, TORCH_VERSION),
        "triton": (importlib.metadata.version("triton"), TRITON_VERSION),
        "CUDA runtime": (torch.version.cuda, CUDA_VERSION),
    }
    for label, (observed, expected) in versions.items():
        if observed != expected:
            raise SystemExit(f"requires {label} {expected}, found {observed}")
    if not torch.cuda.is_available():
        raise SystemExit("the sealed MTP oracle requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability != GPU_COMPUTE_CAPABILITY:
        raise SystemExit(
            "requires the frozen compute capability "
            f"{GPU_COMPUTE_CAPABILITY}, found {capability}"
        )
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != GPU_NAME:
        raise SystemExit(f"requires GPU {GPU_NAME}, found {gpu_name}")
    properties = torch.cuda.get_device_properties(0)
    if (
        properties.multi_processor_count != GPU_SM_COUNT
        or properties.total_memory != GPU_MEMORY_BYTES
    ):
        raise SystemExit(
            "GPU geometry mismatch: expected "
            f"{GPU_SM_COUNT} SMs/{GPU_MEMORY_BYTES} bytes, found "
            f"{properties.multi_processor_count} SMs/{properties.total_memory} bytes"
        )
    if torch.version.git_version != TORCH_GIT_REVISION:
        raise SystemExit(
            f"requires torch revision {TORCH_GIT_REVISION}, "
            f"found {torch.version.git_version}"
        )
    driver = driver_identity()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "gpu_name": gpu_name,
        "gpu_compute_capability": list(capability),
        "gpu_sm_count": properties.multi_processor_count,
        "gpu_memory_bytes": properties.total_memory,
        "python": {
            "version": ".".join(str(part) for part in sys.version_info[:3]),
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
            "platform": sysconfig.get_platform(),
        },
        "torch_git_revision": torch.version.git_version,
        "compiler_sha256": compiler_identity(),
        "triton_compiler_controls": triton_compiler_controls(),
        "inherited_runtime_override_policy": {
            "rejected_prefixes": ["TRITON_"],
            "rejected_exact": sorted(_FORBIDDEN_INHERITED_RUNTIME_VARIABLES),
            "normalized": {
                "CC": "/usr/bin/gcc",
                "cache_roots": "fresh-temporary-directories",
            },
        },
        **driver,
        "installed_distributions": {
            name: distribution_identity(name)
            for name in (
                "blake3",
                "numpy",
                "safetensors",
                "torch",
                "transformers",
                "triton",
                "vllm",
            )
        },
    }


def execute(snapshot: Path, vllm_source: Path) -> dict:
    runtime_identity = verify_runtime()
    source_paths, source_checkout = verify_sources(vllm_source)
    config = make_vllm_config(snapshot, ATTENTION_BACKEND)
    proposer_config = make_proposer_config(snapshot, ATTENTION_BACKEND)
    head = matrix(13, VOCAB, HIDDEN)
    fd, store_path = tempfile.mkstemp(prefix="qwen35-vllm-oracle-")
    os.close(fd)
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{store_path}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            proposer = EagleProposer(proposer_config, torch.device("cuda"))
            with torch.device("cuda"), set_default_torch_dtype(torch.float32):
                target = Qwen3_5ForCausalLM(
                    vllm_config=config, prefix="target"
                ).eval()
                mtp = Qwen3_5MTP(vllm_config=config, prefix="draft").eval()

            target_loaded = target.load_weights(
                sorted(
                    [
                        *[
                            (f"model.{name}", weight)
                            for name, weight in target_weights()
                        ],
                        ("lm_head.weight", head),
                    ]
                )
            )
            mtp_loaded = mtp.load_weights(
                sorted(
                    [
                        ("mtp.embed_tokens.weight", matrix(0, VOCAB, HIDDEN)),
                        *[
                            (f"mtp.{name}", weight)
                            for name, weight in mtp_weights()
                        ],
                        ("lm_head.weight", head),
                    ]
                )
            )
            if len(mtp_weights()) != 15:
                raise AssertionError("MTP fixture must contain exactly 15 HF tensors")
            assert_all_parameters_loaded("target", target, target_loaded)
            assert_all_parameters_loaded("MTP", mtp, mtp_loaded)
            target_expected = expected_target_parameters(target)
            mtp_expected = expected_mtp_parameters(mtp)
            assert_exact_parameters("target", target, target_expected)
            assert_exact_parameters("MTP", mtp, mtp_expected)
            process_weights_after_loading(
                target, config.model_config, torch.device("cuda")
            )
            process_weights_after_loading(
                mtp, config.model_config, torch.device("cuda")
            )
            assert_exact_parameters("target post-load", target, target_expected)
            assert_exact_parameters("MTP post-load", mtp, mtp_expected)

            target_runtime = RuntimeAttention(
                target.model.layers[0].self_attn.attn, config
            )
            mtp_runtime = RuntimeAttention(mtp.model.layers[0].self_attn.attn, config)
            bind_kv_cache(
                {
                    target_runtime.name: target_runtime.cache,
                    mtp_runtime.name: mtp_runtime.cache,
                },
                config.compilation_config.static_forward_context,
                [],
            )

            def target_step(ids: list[int], positions: list[int], seq_len: int):
                common = common_metadata(
                    seq_len=seq_len,
                    query_len=len(ids),
                    device=torch.device("cuda"),
                )
                metadata = target_runtime.metadata(common, drafting=False)
                with torch.inference_mode(), set_forward_context(
                    {target_runtime.name: metadata},
                    config,
                    num_tokens=len(ids),
                    slot_mapping={target_runtime.name: common.slot_mapping},
                ):
                    hidden = target(
                        torch.tensor(ids, dtype=torch.long, device="cuda"),
                        torch.tensor(positions, dtype=torch.long, device="cuda"),
                    )
                logits = target.compute_logits(hidden)
                if logits is None:
                    raise AssertionError("target rank did not produce logits")
                logits = logits[-1]
                return hidden, logits

            def mtp_step(
                target_ids: list[int],
                positions: list[int],
                target_hidden: torch.Tensor,
                sampled_next: int,
                seq_len: int,
            ):
                common = common_metadata(
                    seq_len=seq_len,
                    query_len=len(target_ids),
                    device=torch.device("cuda"),
                )
                target_id_tensor = torch.tensor(
                    target_ids, dtype=torch.int32, device="cuda"
                )
                target_position_tensor = torch.tensor(
                    positions, dtype=torch.int64, device="cuda"
                ).repeat(3, 1)
                num_tokens, sample_indices, draft_common = (
                    proposer.set_inputs_first_pass(
                        target_token_ids=target_id_tensor,
                        next_token_ids=torch.tensor(
                            [sampled_next], dtype=torch.int32, device="cuda"
                        ),
                        target_positions=target_position_tensor,
                        target_hidden_states=target_hidden,
                        token_indices_to_sample=None,
                        cad=common,
                        num_rejected_tokens_gpu=None,
                    )
                )
                if num_tokens != len(target_ids):
                    raise AssertionError("official proposer changed transaction length")
                if sample_indices.tolist() != [len(target_ids) - 1]:
                    raise AssertionError(
                        f"unexpected official proposer sample indices {sample_indices}"
                    )
                official_ids = proposer.input_ids[:num_tokens].clone()
                official_positions = proposer._get_positions(num_tokens).clone()
                expected_positions = target_position_tensor
                if not torch.equal(official_positions, expected_positions):
                    raise AssertionError("official proposer changed target positions")
                official_hidden = proposer.hidden_states[:num_tokens].clone()
                if not torch.equal(official_hidden, target_hidden):
                    raise AssertionError("official proposer changed target hidden states")
                # Both trace rows are first-pass proposals from separate target
                # scheduler iterations, so their official draft index is zero.
                metadata = mtp_runtime.metadata(
                    draft_common, drafting=True, draft_index=0
                )
                with torch.inference_mode(), set_forward_context(
                    {mtp_runtime.name: metadata},
                    config,
                    num_tokens=num_tokens,
                    slot_mapping={mtp_runtime.name: draft_common.slot_mapping},
                ):
                    hidden = mtp(
                        official_ids,
                        official_positions,
                        official_hidden,
                    )
                logits = mtp.compute_logits(hidden)
                if logits is None:
                    raise AssertionError("MTP rank did not produce logits")
                logits = logits[-1]
                keys, values = mtp_runtime.canonical_kv(seq_len)
                return (
                    hidden,
                    logits,
                    keys.clone(),
                    values.clone(),
                    official_ids.tolist(),
                    official_positions.tolist(),
                )

            prompt_ids = [1, 4, 2]
            target_prefill, target_prefill_logits = target_step(
                prompt_ids, [0, 1, 2], 3
            )
            next_id = int(target_prefill_logits.argmax())
            target_decode, target_decode_logits = target_step([next_id], [3], 4)
            next_next_id = int(target_decode_logits.argmax())

            mtp_prefill = mtp_step(
                prompt_ids,
                [0, 1, 2],
                target_prefill,
                next_id,
                3,
            )
            mtp_decode = mtp_step(
                [next_id],
                [3],
                target_decode,
                next_next_id,
                4,
            )
            shifted_prefill_ids = [*prompt_ids[1:], next_id]
            if mtp_prefill[4] != shifted_prefill_ids:
                raise AssertionError(
                    "pinned EagleProposer first-pass shift changed unexpectedly"
                )
            if mtp_decode[4] != [next_next_id]:
                raise AssertionError(
                    "pinned EagleProposer decode shift changed unexpectedly"
                )

            return {
                "oracle": {
                    "vllm_revision": REVISION,
                    "vllm_version": vllm.__version__,
                    "torch_version": torch.__version__,
                    "transformers_version": importlib.metadata.version("transformers"),
                    "backend": ATTENTION_BACKEND,
                    "source_sha256": SOURCE_HASHES,
                    "source_paths": source_paths,
                    "source_checkout": source_checkout,
                    "stubbed_components": [],
                    "evidence_class": "synthetic_fixture",
                    "orchestration": (
                        "direct model execution fed by pinned "
                        "EagleProposer.set_inputs_first_pass and "
                        "build_for_drafting metadata"
                    ),
                    "triton_version": importlib.metadata.version("triton"),
                    "cuda_runtime": torch.version.cuda,
                    **runtime_identity,
                    "triton_dot_input_precision": "tf32",
                    "runtime_cache_policy": "fresh-isolated-no-prior-cache",
                    "triton_jit_cubins_sha256": triton_cubin_hashes(),
                    "loaded_vllm_native_extensions_sha256": (
                        loaded_vllm_native_extensions(vllm_source.resolve())
                    ),
                    "loaded_numeric_libraries_sha256": loaded_numeric_libraries(),
                },
                "fixture": {
                    "profile": "tritium-sealed-mtp-fixture-v1",
                    "hidden": HIDDEN,
                    "intermediate": INTERMEDIATE,
                    "vocab": VOCAB,
                    "query_heads": QUERY_HEADS,
                    "kv_heads": KV_HEADS,
                    "head_dim": HEAD_DIM,
                    "parameter_formula": (
                        "(((17*i + 13*ordinal + 5) % 29) - 14) / 32"
                    ),
                    "target_loaded_params": sorted(target_loaded),
                    "mtp_hf_tensor_count": len(mtp_weights()),
                    "mtp_loaded_params": sorted(mtp_loaded),
                },
                "target": {
                    "prompt_ids": prompt_ids,
                    "prefill_hidden": as_list(target_prefill),
                    "prefill_last_logits": as_list(target_prefill_logits),
                    "sampled_next_id": next_id,
                    "decode_input_id": next_id,
                    "decode_hidden": as_list(target_decode),
                    "decode_last_logits": as_list(target_decode_logits),
                    "sampled_next_next_id": next_next_id,
                },
                "mtp_steps": [
                    {
                        "shifted_ids": mtp_prefill[4],
                        "official_positions": mtp_prefill[5],
                        "draft_index": 0,
                        "target_hidden": as_list(target_prefill),
                        "hidden_states": as_list(mtp_prefill[0]),
                        "last_logits": as_list(mtp_prefill[1]),
                        "cache_keys": as_list(mtp_prefill[2]),
                        "cache_values": as_list(mtp_prefill[3]),
                    },
                    {
                        "shifted_ids": mtp_decode[4],
                        "official_positions": mtp_decode[5],
                        "draft_index": 0,
                        "target_hidden": as_list(target_decode),
                        "hidden_states": as_list(mtp_decode[0]),
                        "last_logits": as_list(mtp_decode[1]),
                        "cache_keys": as_list(mtp_decode[2]),
                        "cache_values": as_list(mtp_decode[3]),
                    },
                ],
                "cache_layout": {
                    "canonical": "[sequence, kv_head, head_dim]",
                    "raw_vllm_logical": "[block, kv_head, sequence, 2*head_dim]",
                    "head_major_negative_control": True,
                    "physical_shape": list(mtp_runtime.cache.shape),
                    "physical_stride": list(mtp_runtime.cache.stride()),
                },
            }
    finally:
        cleanup_dist_env_and_memory()
        try:
            os.unlink(store_path)
        except OSError:
            pass


def implementation_manifest(payload: dict) -> tuple[dict, bytes, bytes]:
    oracle = payload["oracle"]
    fixture = payload["fixture"]
    manifest = {
        "schema": "tritium.qwen35.mtp.oracle-implementation-manifest.v1",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "generator_blake3": blake3.blake3(Path(__file__).read_bytes()).hexdigest(),
        "vllm_revision": oracle["vllm_revision"],
        "vllm_tree": oracle["source_checkout"]["tree"],
        "vllm_source_sha256": oracle["source_sha256"],
        "vllm_native_extensions_sha256": oracle["source_checkout"][
            "native_extensions_sha256"
        ],
        "vllm_version": oracle["vllm_version"],
        "torch_version": oracle["torch_version"],
        "torch_git_revision": oracle["torch_git_revision"],
        "transformers_version": oracle["transformers_version"],
        "triton_version": oracle["triton_version"],
        "cuda_runtime": oracle["cuda_runtime"],
        "cuda_driver_api_version": oracle["cuda_driver_api_version"],
        "nvidia_driver_version": oracle["nvidia_driver_version"],
        "nvml_version": oracle["nvml_version"],
        "driver_library_sha256": oracle["driver_library_sha256"],
        "gpu_name": oracle["gpu_name"],
        "gpu_compute_capability": oracle["gpu_compute_capability"],
        "gpu_sm_count": oracle["gpu_sm_count"],
        "gpu_memory_bytes": oracle["gpu_memory_bytes"],
        "python": oracle["python"],
        "compiler_sha256": oracle["compiler_sha256"],
        "triton_compiler_controls": oracle["triton_compiler_controls"],
        "inherited_runtime_override_policy": oracle[
            "inherited_runtime_override_policy"
        ],
        "installed_distributions": oracle["installed_distributions"],
        "attention_backend": oracle["backend"],
        "orchestration": oracle["orchestration"],
        "runtime_cache_policy": oracle["runtime_cache_policy"],
        "triton_dot_input_precision": oracle["triton_dot_input_precision"],
        "triton_jit_cubins_sha256": oracle["triton_jit_cubins_sha256"],
        "loaded_vllm_native_extensions_sha256": oracle[
            "loaded_vllm_native_extensions_sha256"
        ],
        "loaded_numeric_libraries_sha256": oracle[
            "loaded_numeric_libraries_sha256"
        ],
        "numeric_profile": "fp32-storage-tf32-attention-absolute-2e-3",
        "coverage_profile": "fixture-prefill-decode",
        "evidence_class": "synthetic-fixture",
        "cache_layout": payload["cache_layout"]["canonical"],
        "fixture_profile": fixture["profile"],
        "fixture_geometry": {
            key: fixture[key]
            for key in (
                "hidden",
                "intermediate",
                "vocab",
                "query_heads",
                "kv_heads",
                "head_dim",
            )
        },
        "parameter_formula": fixture["parameter_formula"],
        "source_model_id": FIXTURE_SOURCE_MODEL_ID,
        "source_config_digest": FIXTURE_SOURCE_CONFIG_DIGEST,
    }
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_id = blake3.blake3(
        canonical, derive_key_context=MANIFEST_ID_CONTEXT
    ).digest()
    return manifest, canonical, manifest_id


def append_f32s(body: bytearray, values: list[float]) -> None:
    for value in values:
        body.extend(struct.pack("<f", value))


def oracle_body(payload: dict, manifest_id: bytes) -> tuple[bytes, bytes]:
    if len(manifest_id) != 32:
        raise AssertionError("oracle manifest ID must contain 32 bytes")
    source_model_id = bytes.fromhex(FIXTURE_SOURCE_MODEL_ID)
    source_config_digest = bytes.fromhex(FIXTURE_SOURCE_CONFIG_DIGEST)
    target = payload["target"]
    mtp_steps = payload["mtp_steps"]
    transactions = [
        (
            target["prompt_ids"],
            target["sampled_next_id"],
            target["prefill_hidden"],
            target["prefill_last_logits"],
            mtp_steps[0],
        ),
        (
            [target["decode_input_id"]],
            target["sampled_next_next_id"],
            target["decode_hidden"],
            target["decode_last_logits"],
            mtp_steps[1],
        ),
    ]

    body = bytearray()
    body.extend(b"TRQ35MO\0")
    body.extend(struct.pack("<HHHHHH", 1, 0, 1, 1, 1, 0))
    body.extend(manifest_id)
    body.extend(source_model_id)
    body.extend(source_config_digest)
    body.extend(
        struct.pack(
            "<IIIII",
            ARTIFACT_MAX_CONTEXT,
            HIDDEN,
            VOCAB,
            KV_HEADS * HEAD_DIM,
            len(transactions),
        )
    )
    if len(body) != 136:
        raise AssertionError(f"wrong fixed body header size: {len(body)}")

    cumulative_tokens = 0
    for token_ids, sampled_next, target_hidden, target_logits, mtp in transactions:
        body.extend(struct.pack("<I", len(token_ids)))
        for token_id in token_ids:
            body.extend(struct.pack("<I", token_id))
        body.extend(struct.pack("<I", sampled_next))
        cumulative_tokens += len(token_ids)
        expected_lengths = {
            "target hidden": (target_hidden, len(token_ids) * HIDDEN),
            "target logits": (target_logits, VOCAB),
            "MTP hidden": (mtp["hidden_states"], len(token_ids) * HIDDEN),
            "logits": (mtp["last_logits"], VOCAB),
            "keys": (mtp["cache_keys"], cumulative_tokens * KV_HEADS * HEAD_DIM),
            "values": (
                mtp["cache_values"],
                cumulative_tokens * KV_HEADS * HEAD_DIM,
            ),
        }
        for label, (values, expected) in expected_lengths.items():
            if len(values) != expected:
                raise AssertionError(
                    f"{label} has {len(values)} lanes, expected {expected}"
                )
            append_f32s(body, values)

    body_bytes = bytes(body)
    body_id = blake3.blake3(
        body_bytes, derive_key_context=BODY_ID_CONTEXT
    ).digest()
    return body_bytes, body_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-source",
        type=Path,
        required=True,
        help="clean vLLM checkout at the exact pinned revision",
    )
    parser.add_argument(
        "--identity-bin",
        type=Path,
        default=Path("target/debug/examples/qwen35_hf_identity"),
        help="built Tritium source-identity helper",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--output-body",
        type=Path,
        required=True,
        help="canonical TRQ35MO v1 artifact path",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="canonical oracle implementation manifest path",
    )
    args = parser.parse_args()
    if not args.vllm_source.is_dir():
        raise SystemExit(f"vLLM source does not exist: {args.vllm_source}")

    with tempfile.TemporaryDirectory(prefix="tritium-qwen35-mtp-source-") as temp:
        fixture_dir = Path(temp) / "source"
        write_fixture_source(fixture_dir)
        identity = verify_fixture_identity(fixture_dir, args.identity_bin)
        payload = execute(fixture_dir, args.vllm_source)
        payload["fixture"]["source_identity"] = identity

    manifest, canonical_manifest, manifest_id = implementation_manifest(payload)
    body, body_id = oracle_body(payload, manifest_id)
    payload["oracle"]["implementation_manifest_id"] = manifest_id.hex()
    payload["artifact"] = {
        "body_id": body_id.hex(),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_bytes": len(body),
        "source_model_id": FIXTURE_SOURCE_MODEL_ID,
        "source_config_digest": FIXTURE_SOURCE_CONFIG_DIGEST,
        "max_context": ARTIFACT_MAX_CONTEXT,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    for path in (args.output_json, args.output_body, args.output_manifest):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json is not None:
        args.output_json.write_text(rendered)
    args.output_body.write_bytes(body)
    args.output_manifest.write_bytes(canonical_manifest + b"\n")
    print(
        json.dumps(
            {
                "body_id": body_id.hex(),
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "body_bytes": len(body),
                "implementation_manifest_id": manifest_id.hex(),
                "manifest": manifest,
                "output_json": str(args.output_json) if args.output_json else None,
                "output_body": str(args.output_body),
                "output_manifest": str(args.output_manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
