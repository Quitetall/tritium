# Web training payload V1

`tritium.web_training_payload` V1 initializes a compiled browser-training
session without an object graph or host-endian ambiguity. It contains exactly
one entry for every canonical parameter owner and optimizer-state owner in the
compiled recipe. Batch, gradient, activation, result and internal buffers are
not serialized. The decoder zero-initializes them, except buffers marked with
`backwardInitialization: "one"`.

All integers are unsigned little-endian. Maximum complete payload size is
64 MiB.

| Offset | Width | Field |
| ---: | ---: | --- |
| 0 | 8 | ASCII `TRWEBP1` followed by NUL |
| 8 | 4 | schema version, exactly `1` |
| 12 | 4 | entry count, nonzero |
| 16 | 4 | body byte length |
| 20 | 4 | reserved, exactly zero |
| 24 | 32 | BLAKE3-256 of body bytes |
| 56 | variable | canonical entry body |

Each body entry contains:

| Width | Field |
| ---: | --- |
| 2 | tensor-name UTF-8 byte length, nonzero |
| 1 | dtype: `0=f32`, `1=u32`, `2=bytes` |
| 1 | flags, exactly zero |
| 4 | tensor payload byte length |
| variable | canonical UTF-8 tensor name |
| variable | tensor payload bytes |

Entries are strictly increasing by unsigned UTF-8 byte sequence. F32 payloads
store raw IEEE-754 binary32 lanes; u32 and f32 lanes are little-endian. Names,
dtypes and byte lengths must match compiled owner buffers exactly. Aliases never
receive entries or allocations. Unknown, missing, duplicate, unordered,
noncanonical, corrupt or trailing data fails before persistent state commits.
