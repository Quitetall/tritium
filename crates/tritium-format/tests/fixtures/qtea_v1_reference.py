"""Independent synthetic V1 byte fixture from the ADR0155 field contract.

Only stdlib struct and Python blake3; no Tritium, weights, network or file writes.
The trit bytes below are hand-calculated radix3 digits for eight repeating trits.
"""
import struct
import blake3

header = bytearray(384)
struct.pack_into("<8sIIQQQQQ", header, 0, b"TQTEA1\0\0", 1, 384, 466, 4, 2, 2, 0)
recipe = struct.pack("<QdfIIfdf", 2, 0.0, 0.96, 1, 2, 0.007, 1.0, 1e-7)
recipe_digest = blake3.blake3(b"qtea.cpu-tensor.f32.v1" + recipe,
    derive_key_context="tritium qtea tensor recipe v1").digest()
struct.pack_into("<dfIIfdf", header, 64, 0.0, 0.96, 1, 2, 0.007, 1.0, 1e-7)
for offset, value in [(104, 1), (136, 2), (200, 3), (232, 4), (264, 5)]:
    header[offset:offset + 32] = bytes([value]) * 32
header[168:200] = recipe_digest
struct.pack_into("<Q", header, 296, 50)
body = bytes([102, 119])
body += struct.pack("<8f", 1.0, 0.25, 2.0, -0.5, 3.0, 0.0, 4.0, -0.0)
body += struct.pack("<2f2I", 1.0, 1.0, 1, 0)
assert len(body) == 50
prefix = bytes(header) + body
encoded = prefix + blake3.blake3(prefix,
    derive_key_context="tritium qtea tensor wire v1").digest()
assert len(encoded) == 466
hexadecimal = encoded.hex()
print("\n".join(hexadecimal[i:i + 64] for i in range(0, len(hexadecimal), 64)))
