#!/usr/bin/env python3
"""Generate a small REAL ggml-format GGUF with TQ2_0/TQ1_0/F16/F32 tensors using
the official gguf writer, then read it back and print the ground-truth tensor table
so the Rust test can assert against it."""
import sys, numpy as np
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType

OUT = sys.argv[1]

w = GGUFWriter(OUT, "bitnet")
# BitNet-style metadata (drives ModelConfig::from_gguf too)
w.add_uint32("bitnet.block_count", 1)
w.add_uint32("bitnet.embedding_length", 256)
w.add_uint32("bitnet.attention.head_count", 4)
w.add_uint32("bitnet.attention.head_count_kv", 2)
w.add_uint32("bitnet.feed_forward_length", 64)
w.add_uint32("bitnet.context_length", 4096)
w.add_float32("bitnet.rope.freq_base", 500000.0)
w.add_float32("bitnet.attention.layer_norm_rms_epsilon", 1e-5)
w.add_string("general.architecture", "bitnet")

# token_embd: F16 [K=256, N=32]
w.add_tensor("token_embd.weight", np.zeros((32, 256), dtype=np.float16))
# attn_q: TQ2_0, 8 rows, K=256 -> 1 block/row -> byte shape [8, 66]
w.add_tensor("blk.0.attn_q.weight",
             np.zeros((8, 66), dtype=np.uint8),
             raw_dtype=GGMLQuantizationType.TQ2_0)
# ffn_down: TQ1_0, 4 rows, K=256 -> 1 block/row -> byte shape [4, 54]
w.add_tensor("blk.0.ffn_down.weight",
             np.zeros((4, 54), dtype=np.uint8),
             raw_dtype=GGMLQuantizationType.TQ1_0)
# output_norm: F32 [256]
w.add_tensor("output_norm.weight", np.ones(256, dtype=np.float32))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

# Read back with the official reader -> ground truth for the Rust assertions.
r = GGUFReader(OUT)
print("VERSION", r.fields["GGUF.version"].parts[-1][0] if "GGUF.version" in r.fields else "?")
print("ALIGNMENT", r.alignment)
for t in r.tensors:
    print("TENSOR", t.name, "dims", list(t.shape), "ggml_type", int(t.tensor_type), "offset", int(t.data_offset))
