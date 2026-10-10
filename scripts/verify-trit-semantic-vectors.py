#!/usr/bin/env python3
"""Independent CBOR/preimage implementation of ADR 0054 candidate vectors.

Requires cbor2 and blake3 in the selected Python environment. This verifies
synthetic candidate hash bytes, not owner acceptance or release qualification.
"""

import importlib.metadata
import json
import struct

import blake3
import cbor2


def metadata_hash(context, value):
    encoded = cbor2.dumps(value, canonical=True)
    result = blake3.blake3(derive_key_context=context)
    result.update(struct.pack("<Q", len(encoded)))
    result.update(encoded)
    return result


def vectors():
    metadata = {"Additive": {
        "rows": 2, "cols": 4, "tile": 256, "group": 32,
        "planes": {"Uniform": 2},
        "law": {"anchor": "Group", "relation": "Free", "precision": "F32"},
        "basis": "Identity",
    }}
    trits = bytes([2, 1, 0, 2, 0, 1, 2, 0] * 2)
    scales = struct.pack("<ffff", 0.5, 0.5, 0.5, 0.5)
    additive = metadata_hash("tritium.trit.additive.v1", metadata)
    additive.update(struct.pack("<Q", len(trits)))
    additive.update(trits)
    additive.update(struct.pack("<Q", len(scales)))
    additive.update(scales)
    tensor_digest = additive.digest()
    asset_digest = blake3.blake3(b"{}").digest()
    model = metadata_hash("tritium.trit.model.v1", {
        "architecture": "llama",
        "assets": [{"name": name, "content": list(asset_digest)}
                   for name in ["config.json", "tokenizer.json"]],
        "tensors": [{"name": "model.embed_tokens.weight", "content": list(tensor_digest)}],
    })
    dense = metadata_hash("tritium.trit.dense.v1", {"Dense": {"shape": [2, 4], "precision": "F32"}})
    dense.update(struct.pack("<Q", 32))
    dense.update(bytes(32))
    tied_meta = {"Additive": {
        "rows": 2, "cols": 4, "tile": 256, "group": 32,
        "planes": {"Uniform": 3},
        "law": {"anchor": "Group", "relation": {"Tied": {"num": 1, "den": 3}}, "precision": "F16"},
        "basis": {"SignedRht": {"block": 4, "seed": 7, "domain": 9}},
    }}
    tied = metadata_hash("tritium.trit.additive.v1", tied_meta)
    tied_trits = bytes([2, 1, 0, 2, 0, 1, 2, 0] * 3)
    tied_scales = struct.pack("<ee", 0.5, 1.0)
    tied.update(struct.pack("<Q", len(tied_trits)))
    tied.update(tied_trits)
    tied.update(struct.pack("<Q", len(tied_scales)))
    tied.update(tied_scales)
    return {"additive": tensor_digest.hex(), "model": model.hexdigest(),
            "dense": dense.hexdigest(), "tied": tied.hexdigest()}


if __name__ == "__main__":
    actual = vectors()
    expected = {
        "additive": "fc19a72d5206c01988147cd77700b8d6a7f1ae8977c0321d16763220e89b69cd",
        "dense": "b19646f6931fcd0dab8b73d14f624cdc542865cddd9a6a40d9c798727fa7a568",
        "model": "a396a7331dadad1cf59f96df549cf551784ca5a19f5c6e9990c1f66d6fcbaf5f",
        "tied": "54b0d7099eb3e68572e14a45160b666e70e9d945a827d685f0d331bbe599bdf4",
    }
    if actual != expected:
        raise ValueError(f"candidate vector mismatch: {actual}")
    print(json.dumps({"profile": "ADR-0054-proposed", "status": "candidate-vector-PASS", "vectors": actual,
                      "libraries": {name: importlib.metadata.version(name)
                                    for name in ["blake3", "cbor2"]}}, sort_keys=True))
